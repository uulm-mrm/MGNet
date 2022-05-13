import math
from typing import List, Optional

import torch
import torch.nn.functional as F
from mgnet.geometry import Camera

__all__ = ["get_depth_prediction"]


def get_depth_prediction(
    depth_logits: torch.Tensor,
    use_dgc_scaling: bool,
    camera_matrix: torch.Tensor = None,
    real_camera_height: torch.Tensor = None,
    panoptic_seg: torch.Tensor = None,
    road_class_id: int = -1,
    depth_filter_class_ids: List[int] = None,
):
    """
    Post-processing for depth prediction.
    This function performs depth rescaling using the DGC scaling module poposed in
    https://arxiv.org/abs/2004.05560. If a panoptic segmentation is available, the road_class_id
    is used for ground point extraction in the DGC module. Additionally, the depth prediction
    is set to zero for classes defined in depth_filter_class_ids.

    Args:
        depth_logits: A Tensor of shape [1, 1, H, W] of predicted depth.
        use_dgc_scaling: A bool, whether to use DGC rescaling or not.
        camera_matrix: A Tensor of shape [1, 3, 3]. The intrinsic camera matrix.
        real_camera_height: A Tensor of shape [1]. The real camera mounting height over ground.
        panoptic_seg: A Tensor of shape [H, W] of predicted panoptic label.
        road_class_id: An integer, id of the road class in panoptic_seg.
        depth_filter_class_ids: A List of integers, class ids in panoptic_seg, where depth
            prediction is invalid or impossible and thus will be set to zero, e.g. sky class.

    Returns:
        depth_logits: A Tensor of shape [H, W] of predicted depth,
            rescaled to metric depth if use_dgc_scaling is set to true.
        cam_xyz_points: A Tensor of shape [3, H, W] of projected xyz points in camera coordinates.
    """
    cam_xyz_points = None
    if use_dgc_scaling:
        assert camera_matrix is not None, "camera_matrix is necessary for dgc rescaling!"
        assert real_camera_height is not None, "real_camera_height is necessary for dgc rescaling!"
        if panoptic_seg is not None:
            assert (
                road_class_id != -1
            ), "road_class_id is necessary for dgc rescaling using panoptic prediction!"
        # Use DGC module with panoptic prediction for scale factor estimation
        camera_height = real_camera_height.to(depth_logits.device)
        camera = Camera(K=camera_matrix).to(depth_logits.device)
        cam_xyz_points = camera.reconstruct(depth_logits, frame="c")
        scale_factor = _get_scale_recovery(
            cam_xyz_points,
            camera_height,
            ground_mask=(panoptic_seg == road_class_id) if panoptic_seg is not None else None,
        )
        depth_logits *= scale_factor
        cam_xyz_points *= scale_factor
        cam_xyz_points.squeeze_()
    depth_logits.squeeze_()
    if panoptic_seg is not None:
        if depth_filter_class_ids is None:
            depth_filter_class_ids = []
        for class_id in depth_filter_class_ids:
            depth_logits[panoptic_seg == class_id] = 0
            if cam_xyz_points is not None:
                cam_xyz_points[:, panoptic_seg == class_id] = float("nan")

    return depth_logits, cam_xyz_points


def _get_scale_recovery(
    cam_points: torch.Tensor,
    real_cam_height: torch.Tensor,
    ground_mask: Optional[torch.Tensor] = None,
):
    """
    Calculates depth scale factor based on geometric constraints using the following steps:
    1. Calculate surface normals for each xyz point in cam_points.
    2. If no ground_mask is provided, calculate one based on the surface normals.
    3. Estimate a camera height for each ground point and take the median.
    4. Calculate the scale factor based on real_cam_height and the estimated median camera height.

    Args:
        cam_points: A Tensor of shape [3, H, W] of projected xyz points in camera coordinates.
        real_cam_height: A Tensor of shape [1]. The real camera mounting height over ground.
        ground_mask: A boolean Tensor of shape [H, W] derived from the panoptic label or None.

    Returns:
        scale_factor: The estimated scale factor to convert predicted depth to metric depth.
    """
    surface_normal = _get_surface_normal(cam_points)
    if ground_mask is None:
        ground_mask = _get_ground_mask(cam_points, surface_normal)

    cam_heights = (cam_points * surface_normal).sum(1).abs().unsqueeze(1)
    cam_heights_masked = torch.masked_select(cam_heights, ground_mask)
    cam_height = torch.median(cam_heights_masked).unsqueeze(0)

    scale_factor = torch.reciprocal(cam_height).mul_(real_cam_height)

    return scale_factor


def _get_surface_normal(
    cam_points: torch.Tensor,
    nei: int = 1,
):
    """
    Estimate surface normals from xyz points in camera coordinates.
    Derived from https://github.com/zhenheny/LEGO

    Args:
        cam_points: A Tensor of shape [N, 3, H, W] of projected xyz points in camera coordinates.
        nei: An integer, pixel neighborhood to be considered for surface normal calculation.

    Returns:
        normals: A Tensor of shape [N, 3, H, W],
            estimated surface normals corresponding to cam_points.
    """
    cam_points_ctr = cam_points[:, :, nei:-nei, nei:-nei]
    cam_points_x0 = cam_points[:, :, nei:-nei, 0 : -(2 * nei)]
    cam_points_y0 = cam_points[:, :, 0 : -(2 * nei), nei:-nei]
    cam_points_x1 = cam_points[:, :, nei:-nei, 2 * nei :]
    cam_points_y1 = cam_points[:, :, 2 * nei :, nei:-nei]
    cam_points_x0y0 = cam_points[:, :, 0 : -(2 * nei), 0 : -(2 * nei)]
    cam_points_x0y1 = cam_points[:, :, 2 * nei :, 0 : -(2 * nei)]
    cam_points_x1y0 = cam_points[:, :, 0 : -(2 * nei), 2 * nei :]
    cam_points_x1y1 = cam_points[:, :, 2 * nei :, 2 * nei :]

    vector_x0 = cam_points_x0 - cam_points_ctr
    vector_y0 = cam_points_y0 - cam_points_ctr
    vector_x1 = cam_points_x1 - cam_points_ctr
    vector_y1 = cam_points_y1 - cam_points_ctr
    vector_x0y0 = cam_points_x0y0 - cam_points_ctr
    vector_x0y1 = cam_points_x0y1 - cam_points_ctr
    vector_x1y0 = cam_points_x1y0 - cam_points_ctr
    vector_x1y1 = cam_points_x1y1 - cam_points_ctr

    normal_0 = F.normalize(torch.cross(vector_x0, vector_y0, dim=1), dim=1).unsqueeze(0)
    normal_1 = F.normalize(torch.cross(vector_x1, vector_y1, dim=1), dim=1).unsqueeze(0)
    normal_2 = F.normalize(torch.cross(vector_x0y0, vector_x0y1, dim=1), dim=1).unsqueeze(0)
    normal_3 = F.normalize(torch.cross(vector_x1y0, vector_x1y1, dim=1), dim=1).unsqueeze(0)

    normals = torch.cat((normal_0, normal_1, normal_2, normal_3), dim=0).mean(0)
    normals = F.normalize(normals, dim=1)
    normals = F.pad(normals, (nei, nei, nei, nei), "replicate")

    return normals


def _get_ground_mask(
    cam_points: torch.Tensor,
    normal_map: torch.Tensor,
    threshold: int = 5,
):
    """
    Ground mask estimation based on https://arxiv.org/abs/2004.05560.

    Args:
        cam_points: A Tensor of shape [N, 3, H, W] of projected xyz points in camera coordinates.
        normal_map: A Tensor of shape [N, 3, H, W],
            estimated surface normals corresponding to cam_points.
        threshold: An integer, threshold at which a point is considered a ground point.

    Returns:
        ground_mask: A boolean Tensor of shape [N, H, W],
            which is True for estimated ground points and False otherwise.
    """
    b, _, h, w = normal_map.size()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    threshold = math.cos(math.radians(threshold))
    ones, zeros = torch.ones(b, 1, h, w).cuda(), torch.zeros(b, 1, h, w).cuda()
    vertical = torch.cat((zeros, ones, zeros), dim=1)

    cosine_sim = cos(normal_map, vertical).unsqueeze(1)
    vertical_mask = (cosine_sim > threshold) | (cosine_sim < -threshold)

    y = cam_points[:, 1, :, :].unsqueeze(1)
    ground_mask = vertical_mask.masked_fill(y <= 0, False)

    return ground_mask
