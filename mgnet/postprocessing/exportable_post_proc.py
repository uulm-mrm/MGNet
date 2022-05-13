import torch
import torch.nn as nn
from detectron2.data import MetadataCatalog

from .depth_post_proc import _get_surface_normal
from .panoptic_post_proc import _group_instances_and_fuse_logits

__all__ = ["ExportableMGNetPostProcessing"]


class ExportableMGNetPostProcessing(nn.Module):
    """
    Combined panoptic and depth post processing, which is exportable to torch script.
    Some duplicate code with get_depth_prediction is necessary to make the module compile.
    """

    def __init__(self, cfg):
        super().__init__()
        meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        self.data_cfg = torch.tensor(
            [
                len(meta.thing_dataset_id_to_contiguous_id.values()) + 1,
                max(meta.stuff_dataset_id_to_contiguous_id.values()),
                meta.label_divisor,
            ]
        )
        self.road_class_id = next(
            (item["trainId"] for item in meta.categories if item["name"] == "road"),
            None,
        )
        self.depth_filter_class_ids = []
        for cat in meta.categories:
            if cat["name"] in ["ego vehicle", "sky"]:
                self.depth_filter_class_ids.append(cat["trainId"])

    def forward(
        self,
        sem_seg,
        center_heatmap,
        offsets,
        depth_logits,
        inverse_camera_matrix,
        real_camera_height,
    ):
        # Panoptic postprocessing
        sem_seg = sem_seg.squeeze(0).long()
        center_heatmap = center_heatmap.squeeze()
        center_points = torch.nonzero(center_heatmap > 0)

        sem_seg = _group_instances_and_fuse_logits(center_points, offsets, sem_seg, self.data_cfg)

        # Depth postprocessing
        xm = (
            torch.linspace(0, sem_seg.size(2) - 1, sem_seg.size(2), device=torch.device("cuda"))
            .view(1, 1, -1)
            .expand(1, sem_seg.size(1), sem_seg.size(2))
        )
        ym = (
            torch.linspace(0, sem_seg.size(1) - 1, sem_seg.size(1), device=torch.device("cuda"))
            .view(1, -1, 1)
            .expand(1, sem_seg.size(1), sem_seg.size(2))
        )
        grid = torch.stack([xm, ym, torch.ones_like(xm)], dim=1)
        flat_grid = grid.view(depth_logits.size(0), 3, -1)  # [B, 3, HW]
        # Estimate the outward rays in the camera frame
        xnorm = (inverse_camera_matrix.unsqueeze(0).bmm(flat_grid)).view(
            depth_logits.size(0), 3, depth_logits.size(2), depth_logits.size(3)
        )
        # Scale rays to metric depth
        cam_points = xnorm * depth_logits
        ground_mask = sem_seg == self.road_class_id

        surface_normal = _get_surface_normal(cam_points)
        cam_heights = (cam_points * surface_normal).sum(1).abs().unsqueeze(1)
        cam_heights_masked = torch.masked_select(cam_heights, ground_mask)
        cam_height = torch.median(cam_heights_masked).unsqueeze(0)

        scale_factor = torch.reciprocal(cam_height).mul_(real_camera_height)
        depth_logits *= scale_factor
        cam_points *= scale_factor
        for class_id in self.depth_filter_class_ids:
            mask = (sem_seg == class_id).unsqueeze(0)
            depth_logits[mask] = 0

        cam_points = torch.cat([cam_points.squeeze(0), sem_seg.float()], dim=0).permute(1, 2, 0)
        return sem_seg.int(), depth_logits.squeeze(0), cam_points
