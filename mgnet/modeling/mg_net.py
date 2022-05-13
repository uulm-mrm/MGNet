import math
from functools import partial
from typing import Callable, Dict, List

import torch
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import ShapeSpec
from detectron2.modeling import (
    META_ARCH_REGISTRY,
    SEM_SEG_HEADS_REGISTRY,
    Backbone,
    build_backbone,
    build_sem_seg_head,
)
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from mgnet.geometry import inv2depth
from mgnet.postprocessing import (
    get_depth_prediction,
    get_instance_predictions,
    get_panoptic_prediction,
)
from torch import nn
from torch.cuda.amp import custom_fwd
from torch.nn import functional as F

from .layers import GlobalContextModule, MGNetDecoder, MGNetHead, PoseCNN
from .loss import DeepLabCE, MultiViewPhotometricLoss, OhemCE

__all__ = [
    "MGNet",
    "INS_EMBED_HEADS_REGISTRY",
    "build_ins_embed_head",
    "DEPTH_HEADS_REGISTRY",
    "build_depth_head",
    "ExportableMGNet",
]

INS_EMBED_HEADS_REGISTRY = Registry("INS_EMBED_BRANCHES")
INS_EMBED_HEADS_REGISTRY.__doc__ = """
Registry for instance embedding heads, which make instance embedding predictions from feature maps.
"""

DEPTH_HEADS_REGISTRY = Registry("DEPTH_BRANCHES")
DEPTH_HEADS_REGISTRY.__doc__ = """
Registry for depth heads, which make depth predictions from feature maps.
"""


@META_ARCH_REGISTRY.register()
class MGNet(nn.Module):
    """
    MGNet model described in the paper
    https://openaccess.thecvf.com/content/ICCV2021/papers/Schon_MGNet_Monocular_Geometric_Scene_Understanding_for_Autonomous_Driving_ICCV_2021_paper.pdf  # noqa
    """

    @configurable
    def __init__(
        self,
        *,
        size_divisibility: int,
        pixel_mean: List[float],
        pixel_std: List[float],
        backbone: Backbone,
        global_context: nn.Module,
        sem_seg_head: nn.Module,
        ins_embed_head: nn.Module,
        depth_head: nn.Module,
        pose_net: nn.Module,
        with_panoptic: bool,
        with_depth: bool,
        with_uncertainty: bool,
        msc_flip_eval: bool,
        predict_instances: bool,
        instance_post_proc_func: Callable,
        panoptic_post_proc_func: Callable,
        depth_post_proc_func: Callable,
        panoptic_post_proc_threshold: int,
        panoptic_post_proc_nms_kernel: int,
    ):
        super().__init__()
        self.size_divisibility = size_divisibility
        self.register_buffer(
            "pixel_mean", torch.tensor([x / 255.0 for x in pixel_mean]).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std", torch.tensor([x / 255.0 for x in pixel_std]).view(-1, 1, 1), False
        )

        self.backbone = backbone
        self.bb_features = [k for k, v in self.backbone.output_shape().items()]
        self.global_context = global_context
        self.sem_seg_head = sem_seg_head
        self.ins_embed_head = ins_embed_head
        self.depth_head = depth_head
        self.pose_net = pose_net

        self.with_panoptic = with_panoptic
        self.with_depth = with_depth
        self.with_uncertainty = with_uncertainty
        if self.with_uncertainty:
            self.register_parameter(
                "log_vars", torch.nn.Parameter(torch.zeros(5), requires_grad=True)
            )
        self.msc_flip_eval = msc_flip_eval

        self.predict_instances = predict_instances
        self.instance_post_proc_func = instance_post_proc_func
        self.panoptic_post_proc_func = panoptic_post_proc_func
        self.depth_post_proc_func = depth_post_proc_func

        # Used for ExportableMGNet model.
        self.panoptic_post_proc_threshold = panoptic_post_proc_threshold
        self.panoptic_post_proc_nms_kernel = panoptic_post_proc_nms_kernel

    @classmethod
    def from_config(cls, cfg):
        pixel_mean = cfg.MODEL.PIXEL_MEAN
        pixel_std = cfg.MODEL.PIXEL_STD

        backbone = build_backbone(cfg)
        global_context = GlobalContextModule(
            in_channels=[x[1].channels for x in backbone.output_shape().items()][-1],
            out_channels=cfg.MODEL.GCM.GCM_CHANNELS,
            init_method=cfg.MODEL.GCM.INIT_METHOD,
        )

        with_panoptic = cfg.WITH_PANOPTIC
        with_depth = cfg.WITH_DEPTH
        with_uncertainty = cfg.WITH_UNCERTAINTY

        sem_seg_head, ins_embed_head, depth_head, pose_net = None, None, None, None
        if with_panoptic:
            sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
            ins_embed_head = build_ins_embed_head(cfg, backbone.output_shape())
        if with_depth:
            depth_head = build_depth_head(cfg, backbone.output_shape())
            pose_net = PoseCNN(cfg)

        msc_flip_eval = cfg.TEST.MSC_FLIP_EVAL

        predict_instances = cfg.TEST.EVAL_INSTANCE
        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        instance_post_proc_func = None
        if predict_instances:
            instance_post_proc_func = partial(
                get_instance_predictions,
                thing_ids=list(meta.thing_dataset_id_to_contiguous_id.values()),
                label_divisor=meta.label_divisor,
            )

        panoptic_post_proc_func = None
        if with_panoptic:
            # We cannot use partial here due to the torch.cuda.amp.custom_fwd used
            def panoptic_post_proc_func(sem_seg, center_heatmap, offsets):
                return get_panoptic_prediction(
                    sem_seg,
                    center_heatmap,
                    offsets,
                    num_thing_classes=len(meta.thing_dataset_id_to_contiguous_id.values()),
                    last_stuff_id=max(meta.stuff_dataset_id_to_contiguous_id.values()),
                    label_divisor=meta.label_divisor,
                    stuff_area=cfg.MODEL.POST_PROCESSING.STUFF_AREA,
                    void_label=-1,
                    threshold=cfg.MODEL.POST_PROCESSING.CENTER_THRESHOLD,
                    nms_kernel=cfg.MODEL.POST_PROCESSING.NMS_KERNEL,
                )

        depth_post_proc_func = None
        if with_depth:
            road_class_id = next(
                (
                    item["trainId"] * meta.label_divisor
                    for item in meta.categories
                    if item["name"] == "road"
                ),
                None,
            )
            depth_ignore_ids = []
            if with_depth:
                for cat in meta.categories:
                    if cat["name"] in cfg.INPUT.IGNORED_CATEGORIES_IN_DEPTH:
                        depth_ignore_ids.append(cat["trainId"] * meta.label_divisor)
            depth_post_proc_func = partial(
                get_depth_prediction,
                use_dgc_scaling=cfg.MODEL.POST_PROCESSING.USE_DGC_SCALING,
                road_class_id=road_class_id,
                depth_filter_class_ids=depth_ignore_ids,
            )

        return {
            "size_divisibility": cfg.MODEL.SIZE_DIVISIBILITY,
            "pixel_mean": pixel_mean,
            "pixel_std": pixel_std,
            "backbone": backbone,
            "global_context": global_context,
            "sem_seg_head": sem_seg_head,
            "ins_embed_head": ins_embed_head,
            "depth_head": depth_head,
            "pose_net": pose_net,
            "with_panoptic": with_panoptic,
            "with_depth": with_depth,
            "with_uncertainty": with_uncertainty,
            "msc_flip_eval": msc_flip_eval,
            "predict_instances": predict_instances,
            "instance_post_proc_func": instance_post_proc_func,
            "panoptic_post_proc_func": panoptic_post_proc_func,
            "depth_post_proc_func": depth_post_proc_func,
            "panoptic_post_proc_threshold": cfg.MODEL.POST_PROCESSING.CENTER_THRESHOLD,
            "panoptic_post_proc_nms_kernel": cfg.MODEL.POST_PROCESSING.NMS_KERNEL,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in [C, H, W] format
                   * "image_prev": Tensor, previous image in video sequence in [C, H, W] format
                   * "image_next": Tensor, next image in video sequence in [C, H, W] format
                   * "*_orig": Unjittered image, image_prev and image_next for photometric loss calc
                   * "sem_seg": semantic segmentation ground truth
                   * "center": center points heatmap ground truth
                   * "offset": pixel offsets to center points ground truth
                   * "*_weights": pixel-wise loss weight maps for sem_seg, center and offset
                   * "camera_matrix": The [3, 3] intrinsic camera_matrix for image
                   * "reprojection_mask":  bool tensor to mask out pixels in photometric loss
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict is the results for one image. The dict contains the following keys:

                * "panoptic_seg", "sem_seg": see documentation
                    :doc:`/tutorials/models` for the standard output format
                * "depth": Tensor in [H, W] format, pixel-wise depth prediction
                * "instances": available if ``predict_instances is True``. see documentation
                    :doc:`/tutorials/models` for the standard output format
        """
        inputs, outputs, targets = {}, {}, {}
        images = [x["image"].to(self.device).float() / 255.0 for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)
        inputs["image"] = (images.tensor - self.pixel_mean) / self.pixel_std

        # Process images through pose network during training
        if self.training and self.with_depth:
            images_prev = [x["image_prev"].to(self.device).float() / 255.0 for x in batched_inputs]
            images_prev = ImageList.from_tensors(images_prev, self.size_divisibility)
            inputs["image_prev"] = (images_prev.tensor - self.pixel_mean) / self.pixel_std

            images_next = [x["image_next"].to(self.device).float() / 255.0 for x in batched_inputs]
            images_next = ImageList.from_tensors(images_next, self.size_divisibility)
            inputs["image_next"] = (images_next.tensor - self.pixel_mean) / self.pixel_std

            outputs["poses"] = self.pose_net(torch.cat(list(inputs.values()), 1))

        # Process images through MGNet
        if self.msc_flip_eval and not self.training:
            outputs.update(self.forward_multi_scale_flip(inputs["image"]))
        else:
            features = self.backbone(inputs["image"])
            features["global_context"] = self.global_context(features[self.bb_features[-1]])
            if self.with_panoptic:
                outputs["sem_seg"] = self.sem_seg_head(features)
                outputs["center"], outputs["offset"] = self.ins_embed_head(features)
            if self.with_depth:
                outputs["depth"] = self.depth_head(features)

        if self.training:
            # Add panoptic targets
            if self.with_panoptic:
                sem_seg_targets = [x["sem_seg"].to(self.device) for x in batched_inputs]
                sem_seg_targets = ImageList.from_tensors(
                    sem_seg_targets, self.size_divisibility
                ).tensor
                sem_seg_weights = [x["sem_seg_weights"].to(self.device) for x in batched_inputs]
                sem_seg_weights = ImageList.from_tensors(
                    sem_seg_weights, self.size_divisibility
                ).tensor

                center_targets = [x["center"].to(self.device) for x in batched_inputs]
                center_targets = ImageList.from_tensors(
                    center_targets, self.size_divisibility
                ).tensor.unsqueeze(1)
                center_weights = [x["center_weights"].to(self.device) for x in batched_inputs]
                center_weights = ImageList.from_tensors(
                    center_weights, self.size_divisibility
                ).tensor

                offset_targets = [x["offset"].to(self.device) for x in batched_inputs]
                offset_targets = ImageList.from_tensors(
                    offset_targets, self.size_divisibility
                ).tensor
                offset_weights = [x["offset_weights"].to(self.device) for x in batched_inputs]
                offset_weights = ImageList.from_tensors(
                    offset_weights, self.size_divisibility
                ).tensor
                targets.update(
                    {
                        "sem_seg": sem_seg_targets,
                        "sem_seg_weights": sem_seg_weights,
                        "center": center_targets,
                        "center_weights": center_weights,
                        "offset": offset_targets,
                        "offset_weights": offset_weights,
                    }
                )
            # Add depth targets
            if self.with_depth:
                # Add (non color jittered) images to targets for photometric loss calculation
                images_orig = [
                    x["image_orig"].to(self.device).float() / 255.0 for x in batched_inputs
                ]
                images_orig = ImageList.from_tensors(images_orig, self.size_divisibility).tensor
                images_prev_orig = [
                    x["image_prev_orig"].to(self.device).float() / 255.0 for x in batched_inputs
                ]
                images_prev_orig = ImageList.from_tensors(
                    images_prev_orig, self.size_divisibility
                ).tensor
                images_next_orig = [
                    x["image_next_orig"].to(self.device).float() / 255.0 for x in batched_inputs
                ]
                images_next_orig = ImageList.from_tensors(
                    images_next_orig, self.size_divisibility
                ).tensor

                camera_matrices = [x["camera_matrix"] for x in batched_inputs]
                camera_matrices = torch.stack(camera_matrices, dim=0).to(self.device)
                masks = [x["reprojection_mask"].to(self.device) for x in batched_inputs]
                masks = ImageList.from_tensors(masks, self.size_divisibility).tensor.unsqueeze_(1)
                targets.update(
                    {
                        "image_orig": images_orig,
                        "image_prev_orig": images_prev_orig,
                        "image_next_orig": images_next_orig,
                        "camera_matrix": camera_matrices,
                        "reprojection_mask": masks,
                    }
                )

            # Calculate losses
            losses = {}
            if self.with_panoptic:
                losses.update(self.sem_seg_head.losses(outputs, targets))
                losses.update(self.ins_embed_head.losses(outputs, targets))
            if self.with_depth:
                losses.update(self.depth_head.losses(outputs, targets))

            # Multiply loss values with task specific homoscedastic uncertainty
            if self.with_uncertainty:
                idx = 0
                storage = get_event_storage()
                for key, value in losses.items():
                    storage.put_scalar(key + "_raw", value.detach().item())
                    tau = 1.0 if key == "loss_sem_seg" else 0.5
                    losses[key] = (
                        tau * torch.exp(-self.log_vars[idx]) * value + 0.5 * self.log_vars[idx]
                    )
                    storage.put_scalar(
                        key + "_uncertainty", math.exp(self.log_vars[idx].detach().item())
                    )
                    idx = idx + 1
            return losses

        # Post-processing does not support batched inputs, hence process each input separately.
        processed_results = []
        for idx in range(len(batched_inputs)):
            height = batched_inputs[idx].get("height")
            width = batched_inputs[idx].get("width")
            image_size = images.image_sizes[idx]
            if self.with_panoptic:
                r = sem_seg_postprocess(outputs["sem_seg"][idx], image_size, height, width)
                c = sem_seg_postprocess(outputs["center"][idx], image_size, height, width)
                o = sem_seg_postprocess(outputs["offset"][idx], image_size, height, width)
                # Post-processing to get panoptic segmentation.
                panoptic_prediction = self.panoptic_post_proc_func(
                    sem_seg=r.argmax(dim=0, keepdim=True),
                    center_heatmap=c,
                    offsets=o,
                )
                processed_results.append(
                    {"sem_seg": r, "panoptic_seg": (panoptic_prediction, None)}
                )
                if self.predict_instances:
                    # For instance segmentation evaluation. Disabled by default.
                    instances = self.instance_post_proc_func(
                        sem_seg=r,
                        center_heatmap=c,
                        panoptic_image=panoptic_prediction,
                    )
                    if len(instances) > 0:
                        processed_results[-1]["instances"] = Instances.cat(instances)

            if self.with_depth:
                d = sem_seg_postprocess(outputs["depth"][idx], image_size, height, width)

                # Post-processing to get metric depth prediction.
                depth_prediction, xyz_points = self.depth_post_proc_func(
                    depth_logits=d.unsqueeze(0),
                    camera_matrix=batched_inputs[0]["camera_matrix"].unsqueeze(0)
                    if "camera_matrix" in batched_inputs[0]
                    else None,
                    real_camera_height=batched_inputs[0]["camera_height"]
                    if "camera_height" in batched_inputs[0]
                    else None,
                    panoptic_seg=processed_results[-1]["panoptic_seg"][0]
                    if self.with_panoptic
                    else None,
                )
                if self.with_panoptic:
                    processed_results[-1]["depth"] = (depth_prediction, xyz_points)
                else:
                    processed_results.append({"depth": (depth_prediction, xyz_points)})

        return processed_results

    def forward_multi_scale_flip(self, norm_images, scales=None, flip=True):
        """
        Process norm_images through MGNet by
        1. augmenting norm_images using multiple scales and horizontal flipping
        2. averaging the raw predictions from each augmented forward pass.

        Args:
            norm_images: Tensor of shape [N, 3, H, W], normalized input image batch
            scales: List of float values, scale factors to augment norm_images
            flip: bool, if true, the horizontally flipped image is processed for each scale factor
                in addition to the non-flipped image

        Returns:
            output dict with averaged sem_seg, center, offset and depth predictions
        """
        if scales is None:
            scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        if flip:
            flip_range = 2
        else:
            flip_range = 1

        average_r, average_c, average_o, average_d = None, None, None, None
        for scale in scales:
            x = F.interpolate(norm_images, scale_factor=scale, mode="bilinear", align_corners=True)

            for flip_idx in range(flip_range):
                if flip_idx:
                    x = torch.flip(x, dims=(3,))

                features = self.backbone(x)
                features["global_context"] = self.global_context(features[self.bb_features[-1]])

                if self.with_panoptic:
                    sem_seg_logits = self.sem_seg_head.layers(features)
                    center_logits, offset_logits = self.ins_embed_head.layers(features)
                    r = F.interpolate(
                        sem_seg_logits,
                        scale_factor=self.sem_seg_head.common_stride / scale,
                        mode="bilinear",
                        align_corners=True,
                    )
                    r = F.softmax(r, 1)
                    c = F.interpolate(
                        center_logits,
                        scale_factor=self.ins_embed_head.common_stride / scale,
                        mode="bilinear",
                        align_corners=True,
                    )
                    o = (
                        F.interpolate(
                            offset_logits,
                            scale_factor=self.ins_embed_head.common_stride / scale,
                            mode="bilinear",
                            align_corners=True,
                        )
                        * self.ins_embed_head.common_stride
                        / scale
                    )

                    if flip_idx:
                        r = torch.flip(r, dims=(3,))
                        c = torch.flip(c, dims=(3,))
                        o = torch.flip(o, dims=(3,))
                        o[:, 1, :, :] *= -1  # Multiply x-offsets by -1

                    average_r = average_r + r if average_r is not None else r
                    average_c = average_c + c if average_c is not None else c
                    average_o = average_o + o if average_o is not None else o

                if self.with_depth:
                    depth_logits = self.depth_head.layers(features)[0]
                    d = inv2depth(
                        F.interpolate(
                            depth_logits,
                            scale_factor=self.depth_head.common_stride / scale,
                            mode="bilinear",
                            align_corners=True,
                        )
                    )

                    if flip_idx:
                        d = torch.flip(d, dims=(3,))

                    average_d = average_d + d if average_d is not None else d

        if self.with_panoptic:
            average_r = average_r / (flip_range * len(scales))
            average_c = average_c / (flip_range * len(scales))
            average_o = average_o / (flip_range * len(scales))
        if self.with_depth:
            average_d = average_d / (flip_range * len(scales))

        return {"sem_seg": average_r, "center": average_c, "offset": average_o, "depth": average_d}


@SEM_SEG_HEADS_REGISTRY.register()
class MGNetSemSegHead(MGNetDecoder):
    """
    A semantic segmentation head described in :paper:`MGNet`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        common_stride: int,
        arm_channels: List[int],
        refine_channels: List[int],
        ffm_channels: int,
        head_channels: int,
        init_method: str,
        loss_weight: float,
        loss_type: str,
        loss_top_k: float,
        ohem_threshold: float,
        ohem_n_min: int,
        ignore_value: int,
        num_classes: int,
    ):
        super().__init__(
            input_shape=input_shape,
            common_stride=common_stride,
            arm_channels=arm_channels,
            refine_channels=refine_channels,
            ffm_channels=ffm_channels,
            init_method=init_method,
        )

        self.ignore_value = ignore_value
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.decoder_only = num_classes is None

        self.head = MGNetHead(ffm_channels, head_channels, num_classes, init_method)

        if loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_value)
        elif loss_type == "hard_pixel_mining":
            self.loss = DeepLabCE(ignore_label=ignore_value, top_k_percent_pixels=loss_top_k)
        elif loss_type == "ohem":
            self.loss = OhemCE(
                ignore_label=ignore_value, ohem_threshold=ohem_threshold, n_min=ohem_n_min
            )
        else:
            raise ValueError("Unexpected loss type: %s" % loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            common_stride=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            arm_channels=cfg.MODEL.SEM_SEG_HEAD.ARM_CHANNELS,
            refine_channels=cfg.MODEL.SEM_SEG_HEAD.REFINE_CHANNELS,
            ffm_channels=cfg.MODEL.SEM_SEG_HEAD.FFM_CHANNELS,
            head_channels=cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS,
            init_method=cfg.MODEL.SEM_SEG_HEAD.INIT_METHOD,
            loss_weight=cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            loss_type=cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE,
            loss_top_k=cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K,
            ohem_threshold=cfg.MODEL.SEM_SEG_HEAD.OHEM_THRESHOLD,
            ohem_n_min=cfg.MODEL.SEM_SEG_HEAD.OHEM_N_MIN,
            ignore_value=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        )
        return ret

    def forward(self, features):
        y = self.layers(features)
        y = F.interpolate(y, scale_factor=self.common_stride, mode="bilinear", align_corners=True)
        return y

    def layers(self, features):
        y, _ = super().forward(features)
        y = self.head(y)
        return y

    def losses(self, predictions, targets):
        loss = self.loss(predictions["sem_seg"], targets["sem_seg"], targets["sem_seg_weights"])
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


def build_ins_embed_head(cfg, input_shape):
    """
    Build a instance embedding head from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.INS_EMBED_HEAD.NAME
    return INS_EMBED_HEADS_REGISTRY.get(name)(cfg, input_shape)


@INS_EMBED_HEADS_REGISTRY.register()
class MGNetInsEmbedHead(MGNetDecoder):
    """
    A instance embedding head described in :paper:`MGNet`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        common_stride: int,
        arm_channels: List[int],
        refine_channels: List[int],
        ffm_channels: int,
        head_channels: int,
        init_method: str,
        center_loss_weight: float,
        offset_loss_weight: float,
    ):
        super().__init__(
            input_shape=input_shape,
            common_stride=common_stride,
            arm_channels=arm_channels,
            refine_channels=refine_channels,
            ffm_channels=ffm_channels,
            init_method=init_method,
        )

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight

        self.center_head = MGNetHead(ffm_channels, head_channels, 1, init_method)
        self.offset_head = MGNetHead(ffm_channels, head_channels, 2, init_method)

        self.center_loss = nn.MSELoss(reduction="none")
        self.offset_loss = nn.L1Loss(reduction="none")

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            common_stride=cfg.MODEL.INS_EMBED_HEAD.COMMON_STRIDE,
            arm_channels=cfg.MODEL.INS_EMBED_HEAD.ARM_CHANNELS,
            refine_channels=cfg.MODEL.INS_EMBED_HEAD.REFINE_CHANNELS,
            ffm_channels=cfg.MODEL.INS_EMBED_HEAD.FFM_CHANNELS,
            head_channels=cfg.MODEL.INS_EMBED_HEAD.HEAD_CHANNELS,
            init_method=cfg.MODEL.INS_EMBED_HEAD.INIT_METHOD,
            center_loss_weight=cfg.MODEL.INS_EMBED_HEAD.CENTER_LOSS_WEIGHT,
            offset_loss_weight=cfg.MODEL.INS_EMBED_HEAD.OFFSET_LOSS_WEIGHT,
        )
        return ret

    def forward(self, features):
        center, offset = self.layers(features)
        center = F.interpolate(
            center, scale_factor=self.common_stride, mode="bilinear", align_corners=True
        )
        # Multiply offset by common_stride to get scaled pixel offset values
        offset = (
            F.interpolate(
                offset, scale_factor=self.common_stride, mode="bilinear", align_corners=True
            )
            * self.common_stride
        )
        return center, offset

    def layers(self, features):
        y, _ = super().forward(features)
        center = self.center_head(y)
        offset = self.offset_head(y)
        center.sigmoid_()  # Scale center prediction to [0, 1].
        return center, offset

    def losses(self, predictions, targets):
        loss_center = (
            self.center_loss(predictions["center"], targets["center"]) * targets["center_weights"]
        )
        if targets["center_weights"].sum() > 0:
            loss_center = loss_center.sum() / targets["center_weights"].sum()
        else:
            loss_center = loss_center.sum() * 0
        loss_offset = (
            self.offset_loss(predictions["offset"], targets["offset"]) * targets["offset_weights"]
        )
        if targets["offset_weights"].sum() > 0:
            loss_offset = loss_offset.sum() / targets["offset_weights"].sum()
        else:
            loss_offset = loss_offset.sum() * 0
        return {
            "loss_center": loss_center * self.center_loss_weight,
            "loss_offset": loss_offset * self.offset_loss_weight,
        }


def build_depth_head(cfg, input_shape):
    """
    Build a depth head from `cfg.MODEL.DEPTH_HEAD.NAME`.
    """
    name = cfg.MODEL.DEPTH_HEAD.NAME
    return DEPTH_HEADS_REGISTRY.get(name)(cfg, input_shape)


@DEPTH_HEADS_REGISTRY.register()
class MGNetSelfSupervisedDepthHead(MGNetDecoder):
    """
    A depth head described in :paper:`MGNet`.
    """

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        common_stride: int,
        arm_channels: List[int],
        refine_channels: List[int],
        ffm_channels: int,
        head_channels: int,
        init_method: str,
        msc_loss: bool,
        loss: nn.Module,
    ):
        super().__init__(
            input_shape=input_shape,
            common_stride=common_stride,
            arm_channels=arm_channels,
            refine_channels=refine_channels,
            ffm_channels=ffm_channels,
            init_method=init_method,
        )

        self.n = None
        self.msc_loss = msc_loss
        self.loss = loss

        self.heads = []
        head_in_channels = (
            [ffm_channels, arm_channels[1], arm_channels[0]]
            if self.training and self.msc_loss
            else [ffm_channels]
        )
        for head_in_channel in head_in_channels:
            head = MGNetHead(head_in_channel, head_channels, 1, init_method)
            self.heads.append(head)
        self.heads = torch.nn.ModuleList(self.heads)

    @classmethod
    def from_config(cls, cfg, input_shape):
        loss = MultiViewPhotometricLoss(
            ssim_loss_weight=cfg.MODEL.DEPTH_HEAD.SSIM_LOSS_WEIGHT,
            photometric_loss_weight=cfg.MODEL.DEPTH_HEAD.PHOTOMETRIC_LOSS_WEIGHT,
            smoothing_loss_weight=cfg.MODEL.DEPTH_HEAD.SMOOTHING_LOSS_WEIGHT,
            automask_loss=cfg.MODEL.DEPTH_HEAD.AUTOMASK_LOSS,
            photometric_reduce_op=cfg.MODEL.DEPTH_HEAD.PHOTOMETRIC_REDUCE_OP,
            padding_mode=cfg.MODEL.DEPTH_HEAD.PADDING_MODE,
        )

        ret = dict(
            input_shape={
                k: v for k, v in input_shape.items() if k in cfg.MODEL.INS_EMBED_HEAD.IN_FEATURES
            },
            common_stride=cfg.MODEL.DEPTH_HEAD.COMMON_STRIDE,
            arm_channels=cfg.MODEL.DEPTH_HEAD.ARM_CHANNELS,
            refine_channels=cfg.MODEL.DEPTH_HEAD.REFINE_CHANNELS,
            ffm_channels=cfg.MODEL.DEPTH_HEAD.FFM_CHANNELS,
            head_channels=cfg.MODEL.DEPTH_HEAD.HEAD_CHANNELS,
            init_method=cfg.MODEL.DEPTH_HEAD.INIT_METHOD,
            msc_loss=cfg.MODEL.DEPTH_HEAD.MSC_LOSS,
            loss=loss,
        )
        return ret

    def forward(self, features):
        y = self.layers(features)
        # Scale multi-scale depth predictions to same size
        scale_fators = (
            [self.common_stride, self.common_stride * 2, self.common_stride * 4]
            if self.training and self.msc_loss
            else [self.common_stride]
        )
        inv_depths = [
            F.interpolate(x, scale_factor=stride, mode="bilinear", align_corners=True)
            for x, stride in zip(y, scale_fators)
        ]
        if not self.training:
            # Inverse depth prediction during inference
            return inv2depth(inv_depths[0])
        return inv_depths

    def layers(self, features):
        y, msc_features = super().forward(features)
        msc_features = (
            [y, msc_features[1], msc_features[0]] if self.training and self.msc_loss else [y]
        )
        inv_depths = []
        for head, f in zip(self.heads, msc_features):
            y = head(f)
            # Scale depth prediction to [0, 2.].
            # See https://github.com/TRI-ML/packnet-sfm/blob/6e3161f60e7161115813574557761edaffb1b6d1/packnet_sfm/networks/layers/packnet/layers01.py#L98  # noqa
            y = y.sigmoid_() / 0.5
            inv_depths.append(y)
        return inv_depths

    @custom_fwd(cast_inputs=torch.float32)
    def losses(self, predictions, targets):
        return self.loss(predictions, targets)


@META_ARCH_REGISTRY.register()
class ExportableMGNet(MGNet):
    """
    MGNet model used for onnx and TensorRT export.
    forward function excludes non-exportable postprocessing code.
    """

    def forward(self, input_image_tensor):
        input_image_tensor = torch.permute(input_image_tensor, (0, 3, 1, 2))  # NHWC -> NCHW
        input_image_tensor = (input_image_tensor - self.pixel_mean) / self.pixel_std
        features = self.backbone(input_image_tensor)
        features["global_context"] = self.global_context(features[self.bb_features[-1]])

        sem_seg = self.sem_seg_head(features)
        center, offset = self.ins_embed_head(features)
        depth = self.depth_head(features)

        sem_seg = sem_seg.argmax(dim=1, keepdim=True).int()
        nms_padding = (self.panoptic_post_proc_nms_kernel - 1) // 2
        center[center <= self.panoptic_post_proc_threshold] = -1
        center_heatmap_max_pooled = F.max_pool2d(
            center, kernel_size=self.panoptic_post_proc_nms_kernel, stride=1, padding=nms_padding
        )
        center[center != center_heatmap_max_pooled] = -1

        return sem_seg, center, offset, depth
