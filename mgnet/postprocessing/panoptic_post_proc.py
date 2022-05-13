import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd

__all__ = ["get_panoptic_prediction"]


@custom_fwd(cast_inputs=torch.float32)
def get_panoptic_prediction(
    sem_seg: torch.Tensor,
    center_heatmap: torch.Tensor,
    offsets: torch.Tensor,
    num_thing_classes: int,
    last_stuff_id: int,
    label_divisor: int,
    stuff_area: int,
    void_label: int,
    threshold: float = 0.3,
    nms_kernel: int = 7,
):
    """
    Post-processing for panoptic segmentation.
    This function performs the instance grouping and panoptic fusion to merge the semantic and
    instance logits into the final panoptic prediction. The operations are first proposed in the
    DeeperLab paper (https://arxiv.org/abs/1902.05093) and adopted by MGNet.

    Args:
        sem_seg: A Tensor of shape [1, H, W] of predicted semantic label.
        center_heatmap: A Tensor of shape [1, H, W] of raw center heatmap output.
        offsets: A Tensor of shape [2, H, W] of raw offset output. The order of
            second dim is (offset_y, offset_x).
        num_thing_classes: An integer, number of thing classes in the dataset.
        last_stuff_id: An integer, last train_id of stuff classes in the dataset.
        label_divisor: An integer, used to convert panoptic id =
            semantic id * label_divisor + instance_id.
        stuff_area: An integer, remove stuff whose area is less than stuff_area.
        void_label: An integer, indicates the region has no confident prediction.
        threshold: A float, threshold applied to center heatmap score.
        nms_kernel: An integer, NMS max pooling kernel size.
    Returns:
        A Tensor of shape [H, W], int64.
    """
    if sem_seg.dim() != 3 and sem_seg.size(0) != 1:
        raise ValueError("Semantic prediction with un-supported shape: {}.".format(sem_seg.size()))
    if center_heatmap.dim() != 3:
        raise ValueError(
            "Center prediction with un-supported dimension: {}.".format(center_heatmap.dim())
        )
    if offsets.dim() != 3:
        raise ValueError("Offset prediction with un-supported dimension: {}.".format(offsets.dim()))

    center_heatmap = F.threshold(center_heatmap, threshold, -1)
    nms_padding = (nms_kernel - 1) // 2
    center_heatmap_max_pooled = F.max_pool2d(
        center_heatmap, kernel_size=nms_kernel, stride=1, padding=nms_padding
    )
    center_heatmap[center_heatmap != center_heatmap_max_pooled] = -1
    center_heatmap = center_heatmap.squeeze()
    center_points = torch.nonzero(center_heatmap > 0)

    meta_tensor = torch.tensor([num_thing_classes + 1, last_stuff_id, label_divisor])
    panoptic = _group_instances_and_fuse_logits(center_points, offsets, sem_seg, meta_tensor)

    for k in range(last_stuff_id + 1):
        if panoptic[panoptic == k].shape[0] < stuff_area:
            panoptic[panoptic == k] = void_label

    mask = (panoptic < label_divisor) & (panoptic != void_label)
    panoptic[mask] = panoptic[mask] * label_divisor

    return panoptic.squeeze_(0)


def _group_instances_and_fuse_logits(
    center_points: torch.Tensor,
    offsets: torch.Tensor,
    sem_seg: torch.Tensor,
    meta: torch.Tensor,
):
    """
    Group center_points and offsets with semantic logits to produce a fused panoptic prediction.

    Args:
        center_points: A Tensor of shape [N, 2] of predicted center key-points.
            N is the number of key-points.
        offsets: A Tensor of shape [2, H, W] of raw offset output.
            The order of second dim is (offset_y, offset_x).
        sem_seg: A Tensor of shape [1, H, W] of predicted semantic label.
        meta: A Tensor of shape [3].
            Dataset specific metadata [num_thing_classes + 1, last_stuff_id, label_divisor].
    Returns:
        A Tensor of shape [1, H, W], int64.
    """
    if center_points.size(0) == 0:
        return sem_seg

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
    xy = torch.cat((ym, xm), dim=0)
    offsets = offsets.squeeze()
    offsets += xy

    stuff_mask = sem_seg <= meta[1]
    offsets.masked_fill_(
        stuff_mask.repeat(2, 1, 1), 65535
    )  # We use 65535 as ignore_value in offsets
    cluster = offsets[0, :, :].long()
    # Get final (x, y) values of instance offset prediction
    offsets = offsets[offsets != 65535].view([2, -1]).T

    if offsets.size(0) == 0:
        return sem_seg

    # Cluster offset pixel to their closest predicted center pixel
    center_points = center_points.unsqueeze(1)
    offsets = offsets.unsqueeze(0)
    cluster[cluster != 65535] = 1 + torch.norm(center_points - offsets, dim=2, p=2).argmin(dim=0)
    cluster[cluster == 65535] = 0

    # Calculate class ids for each cluster based on class voting
    num_instances = cluster.max()
    bins = sem_seg.squeeze() - meta[1]
    bins.masked_fill_(bins < 0, 0)
    bins.masked_fill_(cluster == 0, 0)

    bins[bins != 0] += (cluster[bins != 0] - 1) * meta[0]
    bins = torch.bincount(bins[bins != 0], minlength=meta[0] * num_instances).view(
        [int(num_instances.item()), -1]
    )
    class_count = bins.max(1)[1]
    class_ids = torch.arange(0, bins.size(0) + 1).cuda()
    class_ids[1:] += (class_count + meta[1]) * meta[2]

    # Update instance class ids in semantic prediction tensor based on calculated class_ids
    class_ids = torch.index_select(class_ids, 0, cluster[cluster != 0])
    cluster.masked_scatter_(cluster != 0, class_ids).unsqueeze_(0)
    sem_seg.masked_scatter_(sem_seg > meta[1], cluster[cluster != 0])

    return sem_seg
