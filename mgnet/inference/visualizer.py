import cv2
import numpy as np
import torch
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import VisImage, Visualizer
from matplotlib import pyplot as plt

__all__ = ["MGNetVisualizer", "MGNetVideoVisualizer"]


class MGNetVisualizer(Visualizer):
    """
    Extends detectron2.utils.visualizer.Visualizer by functions to visualize instance_heatmaps,
    depth predictions and 3d pointclouds using open3d
    """

    def draw_instance_heatmaps(self, center, offset, center_weights=None, offset_weights=None):
        """
        Draw PanopticDeeplab instance targets as color image using
        direction encoding for offset vectors and heatmaps for center points.
        """
        if isinstance(center, torch.Tensor):
            center = center.numpy()
        if isinstance(offset, torch.Tensor):
            offset = offset.numpy()

        color_map = plt.get_cmap("twilight")
        center_thresh = 0.3

        color_image = np.full((self.output.height, self.output.width, 4), 255, dtype=np.uint8)

        # Draw instance offset vectors
        mask = np.ones_like(offset)
        if offset_weights is not None:
            if isinstance(offset_weights, torch.Tensor):
                offset_weights = offset_weights.numpy()
            mask = np.concatenate([offset_weights, offset_weights], axis=0)
        vectors = offset[mask != 0.0]
        if vectors.size != 0:
            vectors = np.transpose(np.reshape(vectors, (2, vectors.shape[0] // 2)), [1, 0])

            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            if (angles.max() - angles.min()) != 0:
                angles = (angles - angles.min()) / (angles.max() - angles.min())

            color_values = np.squeeze(color_map(angles))
            color_values[:, 3] = 1.0
            color_values *= 255.0
            color_values = color_values.flatten().astype(np.uint8)

            mask = np.transpose(np.concatenate([mask, mask], axis=0), [1, 2, 0])
            color_image[mask != 0.0] = color_values

        # Draw center heatmaps
        if center_weights is not None:
            if isinstance(center_weights, torch.Tensor):
                center_weights = center_weights.numpy()
            center[center_weights[0, :, :] == 0.0] = 0.0
        color_image[center > center_thresh] = [255, 0, 0, 255]
        color_image[center > 0.9] = [0, 255, 0, 255]

        color_image = cv2.resize(
            color_image,
            dsize=None,
            fx=self.output.scale,
            fy=self.output.scale,
            interpolation=cv2.INTER_NEAREST,
        )

        return VisImage(color_image)

    def draw_depth(self, depth):
        """
        Draw depth prediction using matplotlib plasma_r color map.
        """
        if isinstance(depth, torch.Tensor):
            depth = depth.numpy()

        color_map = plt.get_cmap("plasma_r")

        # Clip depth values at 80. meter to exclude large outliers from visualization.
        depth[depth < 0] = 0
        depth[depth > 80.0] = 80.0

        # Normalize depth to range [0,1].
        if (np.max(depth) - np.min(depth)) != 0:
            depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))

        color_image = np.multiply(color_map(depth), 255.0).astype(np.uint8)

        color_image = cv2.resize(
            color_image,
            dsize=None,
            fx=self.output.scale,
            fy=self.output.scale,
            interpolation=cv2.INTER_NEAREST,
        )

        return VisImage(color_image)

    @staticmethod
    def draw_pcl(xyz_points, color_image):
        """
        Convert xyz_points and a color_image from numpy or torch into an open3d PointCloud object.
        Args:
            xyz_points: np.array or torch.tensor of shape [3, H, W],
                where the first dimension contains xyz values for each pixel.
            color_image: np.array or torch.tensor of shape [H, W, 3],
                which can be an input RGB image or a semantic prediction RGB color image.
        Returns:
            open3d.geometry.PointCloud object with 3d point values based on xyz_points and
                color values based on RGB values in color_image.
        """
        if isinstance(xyz_points, torch.Tensor):
            xyz_points = xyz_points.numpy()
        if isinstance(color_image, torch.Tensor):
            color_image = color_image.numpy()

        xyz_points = np.transpose(xyz_points, [1, 2, 0])
        xyz_points = xyz_points.reshape(-1, 3)
        color_image = color_image.reshape(-1, 3).astype(np.float32) / 255.0

        import open3d as open3d

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(xyz_points)
        pcd.colors = open3d.utility.Vector3dVector(color_image)
        pcd = pcd.remove_non_finite_points()
        pcd = pcd.voxel_down_sample(0.25)
        return pcd


class MGNetVideoVisualizer(VideoVisualizer):
    """
    Call MGNetVisualizer functions for each frame.
    """

    def draw_instance_heatmaps(
        self, frame, center, offset, center_weights=None, offset_weights=None
    ):
        frame_visualizer = MGNetVisualizer(frame, self.metadata)
        return frame_visualizer.draw_instance_heatmaps(
            center, offset, center_weights, offset_weights
        )

    def draw_depth(self, frame, depth):
        frame_visualizer = MGNetVisualizer(frame, self.metadata)
        return frame_visualizer.draw_depth(depth)

    def draw_pcl(self, frame, xyz_points, color_image):
        frame_visualizer = MGNetVisualizer(frame, self.metadata)
        return frame_visualizer.draw_pcl(xyz_points, color_image)
