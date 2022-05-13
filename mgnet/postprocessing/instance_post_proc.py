from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.structures import BitMasks, Instances

__all__ = ["get_instance_predictions"]


def get_instance_predictions(
    sem_seg: torch.Tensor,
    center_heatmap: torch.Tensor,
    panoptic_image: torch.Tensor,
    thing_ids: List[int],
    label_divisor: int,
):
    """
    Post-processing to generate instance predictions from panoptic prediction
    based on semantic prediction and instance center heatmap prediction.
    The algorithm is described in the PanopticDeeplab Paper (https://arxiv.org/abs/1911.10194).
    This is only used for evaluation purposes.

    Args:
        sem_seg: A Tensor of shape [1, H, W] of predicted semantic label.
        center_heatmap: A Tensor of shape [1, H, W] of raw center heatmap output.
        panoptic_image: A Tensor of shape [H, W] of panoptic prediction.
        thing_ids: A list of integers, contains all ids of thing classes in the dataset.
        label_divisor: An integer, used to convert
            panoptic id = semantic id * label_divisor + instance_id.
    Returns:
        A List of detectron2 Instances structs.
    """
    instances = []
    semantic_prob = F.softmax(sem_seg, dim=0)
    panoptic_image_cpu = panoptic_image.cpu().numpy()
    for panoptic_label in np.unique(panoptic_image_cpu):
        if panoptic_label == -1:
            continue
        pred_class = panoptic_label // label_divisor
        is_thing = pred_class in thing_ids
        # Get instance segmentation results.
        if is_thing:
            instance = Instances((panoptic_image_cpu.shape[0], panoptic_image_cpu.shape[1]))
            # Evaluation code takes continuous id starting from 0
            instance.pred_classes = torch.tensor([pred_class], device=panoptic_image.device)
            mask = panoptic_image == panoptic_label
            instance.pred_masks = mask.unsqueeze(0)
            # Average semantic probability
            sem_scores = semantic_prob[pred_class, ...]
            sem_scores = torch.mean(sem_scores[mask])
            # Center point probability
            mask_indices = torch.nonzero(mask).float()
            center_y, center_x = (
                torch.mean(mask_indices[:, 0]),
                torch.mean(mask_indices[:, 1]),
            )
            center_scores = center_heatmap[0, int(center_y.item()), int(center_x.item())]
            # Confidence score is semantic prob * center prob.
            instance.scores = torch.tensor(
                [sem_scores * center_scores], device=panoptic_image.device
            )
            # Get bounding boxes
            instance.pred_boxes = BitMasks(instance.pred_masks).get_bounding_boxes()
            instances.append(instance)
    return instances
