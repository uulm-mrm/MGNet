#!/usr/bin/env python3
import os
import subprocess

import onnx
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from inplace_abn import ABN, InPlaceABN, InPlaceABNSync
from mgnet import add_mgnet_config
from mgnet.data import register_all_cityscapes_scene_seg, register_all_kitti_eigen_scene_seg
from mgnet.modeling import ExportableMGNet  # noqa
from mgnet.postprocessing import ExportableMGNetPostProcessing


def convert_inplace_abn(module):
    """
    Convert all InplaceABN/InplaceABNSync in module into exportable ABN.
    Args:
        module (torch.nn.Module):
    Returns:
        If module is InplaceABN/InplaceABNSync, returns a new module.
        Otherwise, in-place convert module and return it.
    Similar to convert_sync_batchnorm in
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
    """
    bn_module = (InPlaceABN, InPlaceABNSync)
    res = module
    if isinstance(module, bn_module):
        res = ABN(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
            module.activation,
            module.activation_param,
        )
    else:
        for name, child in module.named_children():
            new_child = convert_inplace_abn(child)
            if new_child is not child:
                res.add_module(name, new_child)
    return res


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--export-size",
        nargs="+",
        type=int,
        default=[1024, 2048],
        help="Height and width used for export. Default is Cityscapes image size of (1024, 2048)",
    )
    args = parser.parse_args()
    assert len(args.export_size) == 2, "export_size has to be a list with format [H, W]!"
    assert "MODEL.WEIGHTS" in args.opts, "MODEL.WEIGHTS have to be passed as opt for onnx export!"
    cfg = get_cfg()
    add_mgnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.META_ARCHITECTURE = "ExportableMGNet"
    cfg.freeze()
    setup_logger()

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_cityscapes_scene_seg(_root, pseudo_label_generation=True)
    register_all_kitti_eigen_scene_seg(_root, pseudo_label_generation=True)

    model = build_model(cfg)
    convert_inplace_abn(model)
    model.to(torch.device(cfg.MODEL.DEVICE))
    model.eval()
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )

    # Build and export postprocessing module
    postprocessing_model = ExportableMGNetPostProcessing(cfg)
    scripted_postprocessing_model = torch.jit.script(postprocessing_model)
    script_file = os.path.join(os.path.dirname(cfg.MODEL.WEIGHTS), "postprocessing.pt")
    scripted_postprocessing_model.save(script_file)

    example_image_input = torch.rand(
        size=(1, args.export_size[0], args.export_size[1], 3),
        dtype=torch.float32,
        device=torch.device(cfg.MODEL.DEVICE),
    )

    with torch.no_grad():
        torch_out = model(example_image_input)

        # Store onnx model inside trained models OUTPUT_DIR
        onnx_model_file = os.path.join(os.path.dirname(cfg.MODEL.WEIGHTS), "model.onnx")

        input_names = ["input_image"]
        output_names = ["semantic", "center", "offset", "depth"]

        torch.onnx.export(
            model,
            example_image_input,
            onnx_model_file,
            input_names=input_names,
            output_names=output_names,
            opset_version=13,
        )

    original_model = onnx.load(onnx_model_file)
    onnx.checker.check_model(original_model)

    print("Converting to TensorRT now...")
    trt_model_file = os.path.join(os.path.dirname(cfg.MODEL.WEIGHTS), "model.plan")
    # Run onnx2trt with all available optimization passes
    bash_cmd = (
        f"onnx2trt {onnx_model_file} -o {trt_model_file} -w 2147500000 -d 16 -b 0 -O "
        f"'eliminate_deadend;"
        f"eliminate_identity;"
        f"eliminate_nop_dropout;"
        f"eliminate_nop_monotone_argmax;"
        f"eliminate_nop_pad;"
        f"eliminate_nop_transpose;"
        f"eliminate_unused_initializer;"
        f"extract_constant_to_initializer;"
        f"fuse_add_bias_into_conv;"
        f"fuse_bn_into_conv;"
        f"fuse_consecutive_concats;"
        f"fuse_consecutive_log_softmax;"
        f"fuse_consecutive_reduce_unsqueeze;"
        f"fuse_consecutive_squeezes;"
        f"fuse_consecutive_transposes;"
        f"fuse_matmul_add_bias_into_gemm;"
        f"fuse_pad_into_conv;"
        f"fuse_transpose_into_gemm;"
        f"lift_lexical_references;"
        f"nop;"
        f"split_init;"
        f"split_predict'"
    )
    output = subprocess.check_output(["bash", "-c", bash_cmd])
    for line in output.splitlines():
        print(line.decode("utf-8"))
