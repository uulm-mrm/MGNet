#!/usr/bin/env python3
import itertools
import os
import subprocess
from datetime import datetime
from importlib import import_module

import torch
import torch.backends.cudnn
import torch.cuda.amp
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.engine.defaults import _try_get_key
from detectron2.evaluation import CityscapesInstanceEvaluator, DatasetEvaluators
from detectron2.evaluation.testing import flatten_results_dict
from detectron2.projects.deeplab import build_lr_scheduler
from detectron2.utils.events import EventStorage, JSONWriter
from mgnet import add_mgnet_config
from mgnet.data import register_all_cityscapes_scene_seg, register_all_kitti_eigen_scene_seg
from mgnet.evaluation import (
    CityscapesSemSegEvaluator,
    DepthEvaluator,
    EvaluationVisualizer,
    PanopticEvaluator,
    TensorboardImageWriter,
)
from mgnet.modeling import MGNet  # noqa
from mgnet.solver import get_mgnet_optimizer_params


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = [
            TensorboardImageWriter(dataset_name, cfg.OUTPUT_DIR, cfg.TEST.EVAL_PERIOD)
        ]
        if cfg.VISUALIZE_EVALUATION:
            evaluator_list.append(EvaluationVisualizer(dataset_name))
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "cityscapes_scene_seg" and cfg.WITH_PANOPTIC:
            evaluator_list.append(PanopticEvaluator(dataset_name, output_folder))
            if cfg.TEST.EVAL_SEMANTIC:
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
            if cfg.TEST.EVAL_INSTANCE:
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        if evaluator_type in ["cityscapes_scene_seg", "kitti_eigen_scene_seg"] and cfg.WITH_DEPTH:
            evaluator_list.append(
                DepthEvaluator(
                    dataset_name,
                    min_depth=cfg.TEST.MIN_DEPTH,
                    max_depth=cfg.TEST.MAX_DEPTH,
                    # If dgc_scaling is disabled, use common gt median scaling instead.
                    use_gt_scale=not cfg.MODEL.POST_PROCESSING.USE_DGC_SCALING,
                    # eigen_crop is used on KITTI-Eigen for comparison with other methods.
                    use_eigen_crop=True if evaluator_type == "kitti_eigen_scene_seg" else False,
                )
            )
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = create_object_by_string(cfg.INPUT.TRAIN_DATASET_MAPPER)(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = create_object_by_string(cfg.INPUT.TEST_DATASET_MAPPER)(cfg, is_train=False)
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_mgnet_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            head_lr_factor=cfg.SOLVER.HEAD_LR_FACTOR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        )

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping for now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_full_model_gradient_clipping(torch.optim.Adam)(
                params, cfg.SOLVER.BASE_LR
            )
        elif optimizer_type == "ADAMW":
            return maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Overwrite default test function with disabled cudnn benchmark and added amp autocast
        """
        torch.backends.cudnn.benchmark = False
        with torch.cuda.amp.autocast(enabled=cfg.TEST.AMP.ENABLED):
            results = super().test(cfg, model, evaluators)

        # Restore original setting
        torch.backends.cudnn.benchmark = _try_get_key(
            cfg, "CUDNN_BENCHMARK", "train.cudnn_benchmark", default=False
        )

        return results


def create_object_by_string(class_str):
    try:
        module_path, class_name = class_str.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        raise ImportError(class_str)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_mgnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if cfg.WRITE_OUTPUT_TO_SUBDIR:
        config_file_name = args.config_file.split("/")[-1].replace(".yaml", "")
        cfg.OUTPUT_DIR = os.path.join(
            cfg.OUTPUT_DIR, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + config_file_name
        )
        if args.eval_only:
            cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + "-EvalOnly"
    if cfg.COMMIT_ID == "":
        cfg.COMMIT_ID = (
            subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").split("\n")[0]
        )
    assert cfg.WITH_PANOPTIC or cfg.WITH_DEPTH, "Either panoptic or depth or both must be active"
    cfg.freeze()
    default_setup(cfg, args)

    # cudnn deterministic setting should be enabled when training with manual seed
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.deterministic = _try_get_key(cfg, "CUDNN_DETERMINISTIC", default=False)

    return cfg


def main(args):
    cfg = setup(args)

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_cityscapes_scene_seg(_root)
    register_all_kitti_eigen_scene_seg(_root)

    if args.eval_only:
        with EventStorage(cfg.SOLVER.MAX_ITER) as storage:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
            if res:
                flattened_results = flatten_results_dict(res)
                storage.put_scalars(**flattened_results, smoothing_hint=False)
            writer = JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json"))
            writer.write()
            return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
