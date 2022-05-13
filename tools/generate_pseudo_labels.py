#!/usr/bin/env python3
import os
import shutil

import numpy as np
import torch
import torch.utils.data
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode, LazyConfig, configurable, get_cfg
from detectron2.data import DatasetFromList, MapDataset, MetadataCatalog
from detectron2.data.build import _test_loader_from_config, trivial_batch_collator
from detectron2.data.samplers import InferenceSampler
from detectron2.engine import DefaultTrainer, create_ddp_model, default_argument_parser, launch
from detectron2.modeling import build_model
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from mgnet import add_mgnet_config
from mgnet.data import (
    MGNetTestDatasetMapper,
    register_all_cityscapes_scene_seg,
    register_all_kitti_eigen_scene_seg,
)
from mgnet.modeling import MGNet  # noqa
from PIL import Image
from tqdm import tqdm


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_mgnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    assert cfg.WITH_PANOPTIC, "WITH_PANOPTIC = True is required for pseudo label generation!"
    cfg.freeze()
    return cfg


@configurable(from_config=_test_loader_from_config)
def build_detection_test_loader(
    dataset, *, mapper, sampler=None, num_workers=0, total_batch_size=None
):
    if isinstance(dataset, list):
        dataset = DatasetFromList(dataset, copy=False)
    if mapper is not None:
        dataset = MapDataset(dataset, mapper)
    if sampler is None:
        sampler = InferenceSampler(len(dataset))
    world_size = comm.get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )
    return data_loader


def generate_pseudo_labels(args):
    cfg = setup_cfg(args)
    setup_logger(distributed_rank=comm.get_rank())
    cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_cityscapes_scene_seg(_root, pseudo_label_generation=True)
    register_all_kitti_eigen_scene_seg(_root, pseudo_label_generation=True)

    model = build_model(cfg)
    model = create_ddp_model(model, broadcast_buffers=False)
    model.eval()
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )

    mapper = MGNetTestDatasetMapper(cfg, is_train=False)

    for dataset in cfg.DATASETS.TRAIN:
        meta = MetadataCatalog.get(dataset)

        if comm.is_main_process():
            shutil.rmtree(meta.gt_dir, ignore_errors=True)
            os.makedirs(meta.gt_dir)
        comm.synchronize()

        id_map = np.zeros(256, dtype=np.uint8)
        for cat in meta.categories:
            # Exclude ego car predictions from id map for KITTI pseudo label generation
            if cat["name"] == "ego vehicle" and "kitti" in meta.name:
                continue
            id_map[cat["trainId"]] = cat["id"]

        data_loader = build_detection_test_loader(
            cfg, dataset, mapper=mapper, total_batch_size=cfg.SOLVER.IMS_PER_BATCH
        )

        for idx, inputs in enumerate(tqdm(data_loader)):
            with torch.no_grad():
                outputs = model(inputs)

            for _input, output in zip(inputs, outputs):
                panoptic_prediction = output["panoptic_seg"][0].detach().cpu().numpy()

                # Remap train_ids to ids
                panoptic_prediction[panoptic_prediction % meta.label_divisor == 0] = (
                    panoptic_prediction[panoptic_prediction % meta.label_divisor == 0]
                    // meta.label_divisor
                )
                panoptic_prediction[panoptic_prediction < meta.label_divisor] = id_map[
                    panoptic_prediction[panoptic_prediction < meta.label_divisor]
                ]
                panoptic_prediction[panoptic_prediction >= meta.label_divisor] = (
                    id_map[
                        panoptic_prediction[panoptic_prediction >= meta.label_divisor]
                        // meta.label_divisor
                    ]
                    * meta.label_divisor
                    + panoptic_prediction[panoptic_prediction >= meta.label_divisor]
                    % meta.label_divisor
                )

                output_path = _input["file_name"]
                if "cityscapes" in meta.name:
                    output_path = os.path.basename(output_path)
                    # Add city to output_path
                    output_path = os.path.join(_input["file_name"].split("/")[-2], output_path)
                    # Replace image suffix with gt suffix
                    output_path = output_path.replace("_leftImg8bit", "_gtFine_instanceIds")
                    # Add gt dir to path
                    output_path = os.path.join(meta.gt_dir, output_path)
                elif "kitti" in meta.name:
                    output_path = output_path.replace("image", "label")

                os.makedirs("/".join(output_path.split("/")[:-1]), exist_ok=True)
                Image.fromarray(panoptic_prediction.astype(np.uint16)).save(output_path)

        comm.synchronize()

        if comm.is_main_process():
            # Copy fine labels to pseudo label path
            if args.copy_fine_labels and "cityscapes" in meta.name:

                def ignore_func(root, file_list):
                    return [
                        f
                        for f in file_list
                        if os.path.isfile(os.path.join(root, f))
                        and "_gtFine_instanceIds.png" not in f
                    ]

                # Sometimes, copytree raises an Bad File Descriptor Error when copying for the first
                # time. Hence, retry once
                copy_success = False
                error_count = 0
                while copy_success is False and error_count < 2:
                    try:
                        shutil.copytree(
                            meta.gt_dir.replace("_sequence", ""),
                            meta.gt_dir,
                            ignore=ignore_func,
                            dirs_exist_ok=True,
                        )
                        copy_success = True
                        print("Successfully copied gtFine labels to gtFine_sequence dir")
                    except OSError as e:
                        print(e)
                        print("Exception in shutil.copytree! Retrying...")
                        error_count += 1

            # Convert to COCO-style panoptic segmentation format
            if "cityscapes" in meta.name:
                from datasets.prepare_cityscapes import convert2panoptic

                cityscapes_path = "/".join(meta.gt_dir.split("/")[:-1])
                convert2panoptic(cityscapes_path=cityscapes_path, set_names=["train"])
            elif "kitti" in meta.name:
                from datasets.prepare_kitti_eigen import convert2panoptic

                kitti_path = "/".join(meta.gt_dir.split("/")[:-1])
                convert2panoptic(kitti_path=kitti_path, image_split_file=meta.image_file_list)

            # Write generation config to file
            path = os.path.join("/".join(meta.gt_dir.split("/")[:-1]), "generation_config.yaml")
            if isinstance(cfg, CfgNode):
                with PathManager.open(path, "w") as f:
                    f.write(cfg.dump())
            else:
                LazyConfig.save(cfg, path)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--copy-fine-labels",
        default=True,
        help="If true, it will copy manual annotated gtFine labels into the pseudo label dir.",
    )
    args = parser.parse_args()
    print("Command Line Args:", args)
    metadata = launch(
        generate_pseudo_labels,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
