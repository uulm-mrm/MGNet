# Adapted from packnet-sfm
# https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/utils/depth.py

import torch

from .image import gradient_x, gradient_y

__all__ = ["inv2depth", "calc_smoothness"]


def inv2depth(inv_depth):
    if isinstance(inv_depth, tuple) or isinstance(inv_depth, list):
        return [inv2depth(item) for item in inv_depth]
    else:
        return 1.0 / inv_depth.clamp(min=1e-6)


def calc_smoothness(inv_depths, image, num_scales):
    inv_depths_norm = _inv_depths_normalize(inv_depths)
    inv_depth_gradients_x = [gradient_x(d) for d in inv_depths_norm]
    inv_depth_gradients_y = [gradient_y(d) for d in inv_depths_norm]

    weights_x = torch.exp(-torch.mean(torch.abs(gradient_x(image)), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(gradient_y(image)), 1, keepdim=True))

    # Note: Fix gradient addition
    smoothness_x = [inv_depth_gradients_x[i] * weights_x for i in range(num_scales)]
    smoothness_y = [inv_depth_gradients_y[i] * weights_y for i in range(num_scales)]
    return smoothness_x, smoothness_y


def _inv_depths_normalize(inv_depths):
    """
    Inverse depth normalization
    Important to regularize smoothness to not converge to zero.
    (See Learning Depth from Monocular Videos using Direct Methods
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_Depth_From_CVPR_2018_paper.pdf).
    Parameters
    ----------
    inv_depths : list of torch.Tensor [B,1,H,W]
        Inverse depth maps
    Returns
    -------
    norm_inv_depths : list of torch.Tensor [B,1,H,W]
        Normalized inverse depth maps
    """
    mean_inv_depths = [inv_depth.mean(2, True).mean(3, True) for inv_depth in inv_depths]
    return [
        inv_depth / mean_inv_depth.clamp(min=1e-6)
        for inv_depth, mean_inv_depth in zip(inv_depths, mean_inv_depths)
    ]
