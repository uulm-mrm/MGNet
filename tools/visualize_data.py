#!/usr/bin/env python3
import argparse
import os

import cv2
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import setup_logger
from mgnet import add_mgnet_config
from mgnet.data import (
    MGNetTrainDatasetMapper,
    register_all_cityscapes_scene_seg,
    register_all_kitti_eigen_scene_seg,
)
from mgnet.inference import MGNetVisualizer


def setup(args):
    cfg = get_cfg()
    add_mgnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--config-file", metavar="FILE", required=True, help="path to config file")
    parser.add_argument("--scale", type=float, default=0.5, help="image scale for visualizations")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_cityscapes_scene_seg(_root)
    register_all_kitti_eigen_scene_seg(_root)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    mapper = MGNetTrainDatasetMapper(cfg)
    train_data_loader = build_detection_train_loader(cfg, mapper=mapper)
    for batch in train_data_loader:
        for per_image in batch:
            # Pytorch tensor is in (C, H, W) format
            img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
            img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

            visualizer = MGNetVisualizer(img, metadata=metadata, scale=args.scale)
            if "sem_seg" in per_image:
                vis = visualizer.draw_sem_seg(per_image["sem_seg"], alpha=0.5)
                cv2.imshow("semantic seg", vis.get_image()[:, :, ::-1])
            if (
                "center" in per_image
                and "offset" in per_image
                and "center_weights" in per_image
                and "offset_weights" in per_image
            ):
                vis = visualizer.draw_instance_heatmaps(
                    per_image["center"],
                    per_image["offset"],
                    per_image["center_weights"],
                    per_image["offset_weights"],
                )
                cv2.imshow("instance seg", vis.get_image()[:, :, ::-1])

            k = cv2.waitKey()
            if k == 27 or k == 113:  # Esc or q key to stop
                exit(0)
