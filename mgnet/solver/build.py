from typing import Any, Dict, List, Optional

import torch
from inplace_abn import ABN, InPlaceABN, InPlaceABNSync

__all__ = ["get_mgnet_optimizer_params"]


def get_mgnet_optimizer_params(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: Optional[float] = 0.0,
    weight_decay_norm: Optional[float] = 0.0,
    head_lr_factor: Optional[float] = 1.0,
    weight_decay_bias: Optional[float] = 0.0,
) -> List[Dict[str, Any]]:
    """
    Get optimizer params from MGNet model by iterating over each component of the model.
    Note, that module_list is specific for MGNet and will nor work for other models.

    Args:
        model: MGNet model
        base_lr: learning rate used for each parameter
        weight_decay: weight_decay used for conv and linear layer weights.
        weight_decay_norm: weight_decay used for normalization layer weights and biases.
        head_lr_factor: learning rate factor applied to all head modules in module_list.
        weight_decay_bias: weight_decay used for conv and linear layer biases.

    Returns:
        List[Dict[str, Any]]: params used by torch.optim.Optimizer classes for optimization.
    """
    params: List[Dict[str, Any]] = []
    module_list = [
        "backbone",
        "global_context",
        "sem_seg_head",
        "ins_embed_head",
        "depth_head",
        "pose_net",
    ]

    for module in module_list:
        if hasattr(model, module):
            m = getattr(model, module)
            if m is None:
                continue
            lr = base_lr
            if "head" in module:
                lr *= head_lr_factor
            params.extend(
                get_module_parameters(
                    m,
                    lr,
                    weight_decay,
                    weight_decay_norm,
                    weight_decay_bias,
                )
            )

    if hasattr(model, "log_vars"):
        params.append(dict(params=model.log_vars, weight_decay=0.0, multiply_lr=False))

    return params


def get_module_parameters(
    module: torch.nn.Module,
    lr: Optional[float] = None,
    weight_decay: Optional[float] = None,
    weight_decay_norm: Optional[float] = None,
    weight_decay_bias: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Get optimizer params for one subcomponent of MGNet.

    Args:
        module: Subcomponent of MGNet, e.g. model.backbone
        lr: learning rate used for each parameter within the subcomponent
        weight_decay: weight_decay used for conv and linear layer weights.
        weight_decay_norm: weight_decay used for normalization layer weights and biases.
        weight_decay_bias: weight_decay used for conv and linear layer biases.

    Returns:
        List[Dict[str, Any]]: params used by torch.optim.Optimizer classes for optimization.
    """
    params: List[Dict[str, Any]] = []
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
        # Mapillary ABN types
        ABN,
        InPlaceABN,
        InPlaceABNSync,
    )
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params.append(dict(params=[m.weight], lr=lr, weight_decay=weight_decay))
            if m.bias is not None:
                params.append(dict(params=[m.bias], lr=lr, weight_decay=weight_decay_bias))
        elif isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d, torch.nn.ConvTranspose2d)):
            params.append(dict(params=[m.weight], lr=lr, weight_decay=weight_decay))
            if m.bias is not None:
                params.append(dict(params=[m.bias], lr=lr, weight_decay=weight_decay_bias))
        elif isinstance(m, norm_module_types):
            params.append(dict(params=[m.weight], lr=lr, weight_decay=weight_decay_norm))
            params.append(dict(params=[m.bias], lr=lr, weight_decay=weight_decay_norm))
    return params
