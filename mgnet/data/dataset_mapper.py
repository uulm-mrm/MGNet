import copy
import logging
from typing import Callable, List, Union

import numpy as np
import torch
from detectron2.config import configurable
from detectron2.data import DatasetMapper, MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from panopticapi.utils import rgb2id

from .target_generator import PanopticDeepLabTargetGenerator
from .transform import ColorJitterAug, RandomPadWithCamMatrixAug, ResizeShortestEdgeWithCamMatrixAug

__all__ = ["MGNetTrainDatasetMapper", "MGNetTestDatasetMapper"]


class MGNetTrainDatasetMapper:
    """
    The callable currently does the following:

    1. Read the image from "file_name" and label from "pan_seg_file_name".
       If depth training is enabled, it further reads image_prev and image_next
       as well as camera calibration infos from "calibration_info"
    2. Applies random scale, crop, flip and color jitter transforms to images, label, camera matrix
    3. Prepare data to Tensor and generate training targets
    """

    @configurable
    def __init__(
        self,
        is_train: bool = True,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        color_jitter_augmentation: T.Augmentation,
        image_format: str,
        with_depth: bool,
        panoptic_target_generator: Callable,
        depth_ignore_ids: List[int],
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            color_jitter_augmentation: color_jitter_augmentation which is applied to the image.
                None, if no color_jitter_augmentation is used.
            image_format: an image format supported by :func:`detection_utils.read_image`.
            with_depth: whether to create targets for self-supervised depth training.
            panoptic_target_generator: a callable that takes "panoptic_seg" and
                "segments_info" to generate training targets for the model.
            depth_ignore_ids: a list of category ids which will be ignored in depth training.
                Usually this includes ego vehicle and sky label.
        """
        self.is_train = is_train
        self.augmentations = T.AugmentationList(augmentations)
        self.color_jitter_augmentation = color_jitter_augmentation
        self.image_format = image_format
        self.with_depth = with_depth
        self.panoptic_target_generator = panoptic_target_generator
        self.depth_ignore_ids = depth_ignore_ids

        logger = logging.getLogger(__name__)
        logger.info("Augmentations used in training: " + str(augmentations))

    @classmethod
    def from_config(cls, cfg):
        with_depth = cfg.WITH_DEPTH

        # Define augmentations
        augs = [
            ResizeShortestEdgeWithCamMatrixAug(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                T.RandomCrop(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                )
            )
            if cfg.INPUT.CROP.RANDOM_PAD_TO_CROP_SIZE:
                image_mean_pad_val = np.repeat(np.expand_dims(cfg.MODEL.PIXEL_MEAN, 1), 2, axis=1)
                augs.append(RandomPadWithCamMatrixAug(cfg.INPUT.CROP.SIZE, image_mean_pad_val, 0))
        augs.append(T.RandomFlip())
        color_jitter_aug = None
        if cfg.INPUT.COLOR_JITTER.ENABLED:
            color_jitter_aug = ColorJitterAug(
                brightness=cfg.INPUT.COLOR_JITTER.BRIGHTNESS,
                contrast=cfg.INPUT.COLOR_JITTER.CONTRAST,
                saturation=cfg.INPUT.COLOR_JITTER.SATURATION,
                hue=cfg.INPUT.COLOR_JITTER.HUE,
            )

        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        panoptic_target_generator = PanopticDeepLabTargetGenerator(
            ignore_label=meta.ignore_label,
            thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
            sigma=cfg.INPUT.GAUSSIAN_SIGMA,
            ignore_stuff_in_offset=cfg.INPUT.IGNORE_STUFF_IN_OFFSET,
            small_instance_area=cfg.INPUT.SMALL_INSTANCE_AREA,
            small_instance_weight=cfg.INPUT.SMALL_INSTANCE_WEIGHT,
            ignore_crowd_in_semantic=cfg.INPUT.IGNORE_CROWD_IN_SEMANTIC,
        )

        depth_ignore_ids = []
        if with_depth:
            for cat in meta.categories:
                if cat["name"] in cfg.INPUT.IGNORED_CATEGORIES_IN_DEPTH:
                    depth_ignore_ids.append(cat["trainId"])

        ret = {
            "augmentations": augs,
            "color_jitter_augmentation": color_jitter_aug,
            "image_format": cfg.INPUT.FORMAT,
            "with_depth": with_depth,
            "panoptic_target_generator": panoptic_target_generator,
            "depth_ignore_ids": depth_ignore_ids,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MGNetTrainDatasetMapper should only be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # Load images.
        image_orig = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image_orig)
        image_prev_orig, image_next_orig = None, None
        if self.with_depth:
            image_prev_orig = utils.read_image(
                dataset_dict["prev_img_file_name"], format=self.image_format
            )
            utils.check_image_size(dataset_dict, image_prev_orig)
            image_next_orig = utils.read_image(
                dataset_dict["next_img_file_name"], format=self.image_format
            )
            utils.check_image_size(dataset_dict, image_next_orig)

        # Panoptic label is encoded in RGB image.
        pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")

        # Reuses semantic transform for panoptic labels.
        aug_input = T.AugInput(image_orig, sem_seg=pan_seg_gt)
        tfl = self.augmentations(aug_input)
        image_orig, pan_seg_gt = aug_input.image, aug_input.sem_seg

        # Apply color jitter augmentation separately.
        # Original images will be used for photometric loss calculation.
        color_jitter_tf = None
        if self.color_jitter_augmentation is not None:
            color_jitter_aug_input = T.AugInput(image_orig)
            color_jitter_tf = self.color_jitter_augmentation(color_jitter_aug_input)
            image = color_jitter_aug_input.image
        else:
            image = image_orig

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose([2, 0, 1])))

        # Generates training targets.
        pan_seg_gt = rgb2id(pan_seg_gt)
        panoptic_targets = self.panoptic_target_generator(pan_seg_gt, dataset_dict["segments_info"])
        dataset_dict.update(panoptic_targets)

        if self.with_depth:
            # Apply image augmentations to context images
            image_prev_orig = tfl.apply_image(image_prev_orig)
            image_next_orig = tfl.apply_image(image_next_orig)

            if self.color_jitter_augmentation is not None:
                image_prev = color_jitter_tf.apply_image(image_prev_orig)
                image_next = color_jitter_tf.apply_image(image_next_orig)
            else:
                image_prev = image_prev_orig
                image_next = image_next_orig

            dataset_dict["image_orig"] = torch.as_tensor(
                np.ascontiguousarray(image_orig.transpose([2, 0, 1]))
            )
            dataset_dict["image_prev_orig"] = torch.as_tensor(
                np.ascontiguousarray(image_prev_orig.transpose([2, 0, 1]))
            )
            dataset_dict["image_prev"] = torch.as_tensor(
                np.ascontiguousarray(image_prev.transpose([2, 0, 1]))
            )
            dataset_dict["image_next_orig"] = torch.as_tensor(
                np.ascontiguousarray(image_next_orig.transpose([2, 0, 1]))
            )
            dataset_dict["image_next"] = torch.as_tensor(
                np.ascontiguousarray(image_next.transpose([2, 0, 1]))
            )

            # Generate reprojection_mask to mask out certain semantic classes from photometric loss.
            reprojection_mask = np.ones_like(pan_seg_gt, dtype=np.bool)
            for id in self.depth_ignore_ids:
                reprojection_mask[dataset_dict["sem_seg"] == id] = 0

            # Generate camera matrix targets.
            optical_center = np.array(
                [
                    dataset_dict["calibration_info"]["intrinsic"]["u0"],
                    dataset_dict["calibration_info"]["intrinsic"]["v0"],
                ]
            ).reshape(1, 2)
            focal_length = np.array(
                [
                    dataset_dict["calibration_info"]["intrinsic"]["fx"],
                    dataset_dict["calibration_info"]["intrinsic"]["fy"],
                ]
            ).reshape(1, 2)

            # Apply augmentations
            # Use apply_coords() to augment optical center values.
            optical_center = tfl.apply_coords(optical_center)
            # noinspection PyTypeChecker
            for tf in tfl:
                # apply_reprojection_mask() is defined for PadTransformWithSegmentation and
                # apply_focal() is defined for ResizeTransformWithCamMatrixAug.
                # Other transforms do not affect these targets.
                try:
                    reprojection_mask = tf.apply_reprojection_mask(reprojection_mask)
                except AttributeError:
                    pass
                try:
                    focal_length = tf.apply_focal(focal_length)
                except AttributeError:
                    pass

            # fmt: off
            camera_matrix = np.array([[focal_length[0, 0],                  0, optical_center[0, 0], 0],   # noqa
                                      [                 0, focal_length[0, 1], optical_center[0, 1], 0],   # noqa
                                      [                 0,                  0,                    1, 0],   # noqa
                                      [                 0,                  0,                    0, 1]],  # noqa
                                     dtype=np.float32)
            # fmt: on
            dataset_dict["camera_matrix"] = torch.as_tensor(camera_matrix.astype(np.float32))
            dataset_dict["reprojection_mask"] = torch.as_tensor(reprojection_mask.astype(np.bool))
            dataset_dict["camera_height"] = torch.tensor(
                [dataset_dict["calibration_info"]["extrinsic"]["z"]]
            )

        return dataset_dict


class MGNetTestDatasetMapper(DatasetMapper):
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train is False, "MGNetTestDatasetMapper cannot be used for training!"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        aug_input = T.AugInput(image, sem_seg=None)
        self.augmentations(aug_input)
        image = aug_input.image

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose([2, 0, 1])))

        # Build camera matrix
        fx = dataset_dict["calibration_info"]["intrinsic"]["fx"]
        fy = dataset_dict["calibration_info"]["intrinsic"]["fy"]
        u0 = dataset_dict["calibration_info"]["intrinsic"]["u0"]
        v0 = dataset_dict["calibration_info"]["intrinsic"]["v0"]
        # fmt: off
        dataset_dict["camera_matrix"] = torch.tensor([[fx,  0, u0],  # noqa
                                                      [ 0, fy, v0],  # noqa
                                                      [ 0,  0,  1]],  # noqa
                                                     dtype=torch.float32)
        # fmt: on
        dataset_dict["camera_height"] = torch.tensor(
            [dataset_dict["calibration_info"]["extrinsic"]["z"]]
        )

        # USER: Modify this if you want to keep them for some reason.
        dataset_dict.pop("annotations", None)
        dataset_dict.pop("sem_seg_file_name", None)
        dataset_dict.pop("pan_seg_file_name", None)
        dataset_dict.pop("prev_img_file_name", None)
        dataset_dict.pop("next_img_file_name", None)
        dataset_dict.pop("segments_info", None)

        return dataset_dict
