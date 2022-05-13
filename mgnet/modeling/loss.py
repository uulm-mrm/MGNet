import torch
import torch.nn as nn
import torch.nn.functional as F
from mgnet.geometry import Camera, Pose, calc_smoothness, inv2depth, match_scales, view_synthesis

__all__ = ["DeepLabCE", "OhemCE", "MultiViewPhotometricLoss"]


class DeepLabCE(nn.Module):
    """
    Hard pixel mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Paper: DeeperLab: Single-Shot Image Parser
    Arguments:
        ignore_label: Integer, label to ignore.
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its
            value < 1.0, only compute the loss for the top k percent pixels
            (e.g., the top 20% pixels). This is useful for hard pixel mining.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, top_k_percent_pixels=1.0, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def forward(self, logits, labels, weights=None):
        if weights is None:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        else:
            # Apply per-pixel loss weights.
            pixel_losses = self.criterion(logits, labels) * weights
            pixel_losses = pixel_losses.contiguous().view(-1)
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        return pixel_losses.mean()


class OhemCE(nn.Module):
    """
    Online hard example mining with cross entropy loss, for semantic segmentation.
    This is used in most Pytorch semantic segmentation frameworks.
    Arguments:
        ignore_label: Integer, label to ignore.
        ohem_threshold: Float, the value lies in [0.0, 1.0]. Threshold for hard example selection.
            Below which are predictions with low confidence.
            Hard examples will be pixels of top ``n_min`` loss.
        n_min: Integer, the minimum number of predictions to keep.
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=-1, ohem_threshold=0.7, n_min=100000, weight=None):
        super(OhemCE, self).__init__()
        self.ohem_threshold = -torch.log(torch.tensor(ohem_threshold, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def forward(self, logits, labels, weights=None):
        if weights is None:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        else:
            # Apply per-pixel loss weights.
            pixel_losses = self.criterion(logits, labels) * weights
            pixel_losses = pixel_losses.contiguous().view(-1)

        pixel_losses, _ = torch.sort(pixel_losses, descending=True)
        if pixel_losses[self.n_min] > self.ohem_threshold:
            pixel_losses = pixel_losses[pixel_losses > self.ohem_threshold]
        else:
            pixel_losses = pixel_losses[: self.n_min]

        return pixel_losses.mean()


class MultiViewPhotometricLoss(nn.Module):
    # Adapted from packnet-sfm
    # https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/losses/multiview_photometric_loss.py  # noqa
    def __init__(
        self,
        ssim_loss_weight,
        photometric_loss_weight,
        smoothing_loss_weight,
        automask_loss,
        photometric_reduce_op,
        padding_mode,
    ):
        super().__init__()
        self.n = None
        self.ssim_loss_weight = ssim_loss_weight
        self.photometric_loss_weight = photometric_loss_weight
        self.smoothing_loss_weight = smoothing_loss_weight
        self.automask_loss = automask_loss
        self.photometric_reduce_op = photometric_reduce_op
        self.padding_mode = padding_mode

        # Asserts
        if self.automask_loss:
            assert (
                self.photometric_reduce_op == "min"
            ), "For automasking only the min photometric_reduce_op is supported."

    def forward(self, predictions, targets):
        inv_depths = predictions["depth"]
        pose_results = predictions["poses"]
        self.n = len(inv_depths)

        context = [targets["image_prev_orig"], targets["image_next_orig"]]
        poses = [
            Pose.from_vec(pose_results[:, i].float(), "euler") for i in range(pose_results.shape[1])
        ]
        assert len(context) == len(poses), "Context and poses lists must be of same length"

        camera_matrix = targets["camera_matrix"][:, :3, :3]
        ref_camera_matrix = targets["camera_matrix"][:, :3, :3]

        images = match_scales(targets["image_orig"], inv_depths, self.n)
        depths = [inv2depth(inv_depths[i]) for i in range(self.n)]
        cams = [Camera(K=camera_matrix.float()).to(inv_depths[0].device)]

        # Loop over all reference images
        photometric_losses = [[] for _ in range(self.n)]
        for j, (ref_image, pose) in enumerate(zip(context, poses)):
            # Calculate warped images
            ref_warped = self.warp_ref_image(depths, ref_image, cams, ref_camera_matrix, pose)
            # Calculate and store image loss
            photometric_l = self.calc_photometric_loss(ref_warped, images)
            for i in range(self.n):
                photometric_losses[i].append(photometric_l[i])
            # If using automask
            if self.automask_loss:
                # Calculate and store unwarped image loss
                # All images have the same size, so we only need to calculate this for the first one
                unwarped_image_loss = self.calc_photometric_loss([ref_image], [images[0]])
                for i in range(self.n):
                    photometric_losses[i].append(unwarped_image_loss[0])

        # Calculate reduced photometric loss
        mask = targets["reprojection_mask"] if "reprojection_mask" in targets else None
        photometric_loss = self.reduce_photometric_loss(photometric_losses, mask)
        smoothness_loss = self.calc_smoothness_loss(inv_depths, images, mask)

        return {
            "loss_photometric": photometric_loss * self.photometric_loss_weight,
            "loss_smoothness": smoothness_loss * self.smoothing_loss_weight,
        }

    def warp_ref_image(self, depths, ref_image, cams, ref_camera_matrix, pose):
        """
        Warps a reference image to produce a reconstruction of the original one.
        """
        device = ref_image.get_device()
        ref_cam = Camera(K=ref_camera_matrix.float(), Tcw=pose.to(device))
        assert len(cams) == 1
        # Return warped reference image
        return [
            view_synthesis(ref_image, depths[i], ref_cam, cams[0], padding_mode=self.padding_mode)
            for i in range(self.n)
        ]

    def calc_photometric_loss(self, t_est, images):
        """
        Calculates the photometric loss (L1 + SSIM)
        Parameters
        ----------
        t_est : list of torch.Tensor [B,3,H,W]
            List of warped reference images in multiple scales
        images : list of torch.Tensor [B,3,H,W]
            List of original images in multiple scales

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Photometric loss
        """
        # L1 loss
        l1_loss = [torch.abs(t_est[i] - images[i]) for i in range(len(t_est))]
        # SSIM loss
        if self.ssim_loss_weight > 0.0:
            ssim_loss = [self.ssim(t_est[i], images[i], kernel_size=3) for i in range(len(t_est))]
            # Weighted Sum: alpha * ssim + (1 - alpha) * l1
            photometric_loss = [
                self.ssim_loss_weight * ssim_loss[i].mean(1, True)
                + (1 - self.ssim_loss_weight) * l1_loss[i].mean(1, True)
                for i in range(len(t_est))
            ]
        else:
            photometric_loss = l1_loss
        # Return total photometric loss
        return photometric_loss

    @staticmethod
    def ssim(x, y, kernel_size=3, c1=1e-4, c2=9e-4, stride=1):
        """Calculates the SSIM (Structural Similarity) loss"""
        x, y = F.pad(x, [1, 1, 1, 1], "reflect"), F.pad(y, [1, 1, 1, 1], "reflect")
        mu_x = F.avg_pool2d(x, kernel_size, stride=stride)
        mu_y = F.avg_pool2d(y, kernel_size, stride=stride)

        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = F.avg_pool2d(x.pow(2), kernel_size, stride=stride) - mu_x_sq
        sigma_y = F.avg_pool2d(y.pow(2), kernel_size, stride=stride) - mu_y_sq
        sigma_xy = F.avg_pool2d(x * y, kernel_size, stride=stride) - mu_x_mu_y

        ssim_value = (
            (2 * mu_x_mu_y + c1)
            * (2 * sigma_xy + c2)
            / ((mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2))
        )
        return torch.clamp((1.0 - ssim_value) / 2.0, 0.0, 1.0)

    def reduce_photometric_loss(self, photometric_losses, mask):
        """
        Combine the photometric loss from all context images

        Parameters
        ----------
        photometric_losses : list of torch.Tensor [B,3,H,W]
            Pixel-wise photometric losses from the entire context
        mask: Reprojection mask to mask out padded pixels from augmentation

        Returns
        -------
        photometric_loss : torch.Tensor [1]
            Reduced photometric loss
        """
        if mask is None:
            mask = torch.ones_like(photometric_losses[0][0], dtype=torch.bool)

        # Reduce function
        def reduce_function(losses):
            if self.photometric_reduce_op == "mean":
                return sum([loss[mask].mean() for loss in losses]) / len(losses)
            elif self.photometric_reduce_op == "min":
                return torch.cat(losses, 1).min(1, True)[0][mask].mean()
            else:
                raise NotImplementedError(
                    "Unknown photometric_reduce_op: {}".format(self.photometric_reduce_op)
                )

        # Reduce photometric loss
        photometric_loss = (
            sum([reduce_function(photometric_losses[i]) for i in range(self.n)]) / self.n
        )
        return photometric_loss

    def calc_smoothness_loss(self, inv_depths, images, mask=None):
        """
        Calculates the smoothness loss for inverse depth maps.

        Parameters
        ----------
        inv_depths : list of torch.Tensor [B,1,H,W]
            Predicted inverse depth maps for all scales
        images : list of torch.Tensor [B,3,H,W]
            Original images for all scales
        mask: Reprojection mask to mask out padded pixels from augmentation

        Returns
        -------
        smoothness_loss : torch.Tensor [1]
            Smoothness loss
        """
        if mask is None:
            mask = torch.ones_like(inv_depths[0], dtype=torch.bool)

        # Calculate smoothness gradients
        smoothness_x, smoothness_y = calc_smoothness(inv_depths, images[0], self.n)

        # Calculate smoothness loss
        smoothness_loss = (
            sum(
                [
                    (
                        smoothness_x[i][mask[:, :, :, :-1]].abs().mean()
                        + smoothness_y[i][mask[:, :, :-1, :]].abs().mean()
                    )
                    / 2 ** i
                    for i in range(self.n)
                ]
            )
            / self.n
        )
        return smoothness_loss
