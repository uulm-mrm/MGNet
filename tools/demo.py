#!/usr/bin/env python3
import argparse
import glob
import multiprocessing as mp
import os
import time

import cv2
import open3d
import torch
import tqdm
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import ColorMode
from matplotlib import pyplot as plt
from mgnet import add_mgnet_config
from mgnet.data import register_all_cityscapes_scene_seg, register_all_kitti_eigen_scene_seg
from mgnet.inference import MGNetPredictor, MGNetVideoVisualizer, MGNetVisualizer
from mgnet.modeling import MGNet  # noqa


class VisualizationDemo(object):
    def __init__(self, cfg, calibration_file=None, show_pcl=False, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.show_pcl = show_pcl
        self.instance_mode = instance_mode
        self.predictor = MGNetPredictor(cfg, calibration_file)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in RGB order).

        Returns:
            panoptic_output: the visualized panoptic prediction or None
            depth_output: the visualized depth prediction or None
            pcl_output: open3d.geometry.PointCloud object or None
        """
        predictions = self.predictor(image)
        visualizer = MGNetVisualizer(image, self.metadata, instance_mode=self.instance_mode)
        panoptic_output = None
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            panoptic_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        depth_output = None
        pcl_output = None
        if "depth" in predictions:
            depth = predictions["depth"][0].to(self.cpu_device)
            depth_output = visualizer.draw_depth(depth)
            if self.show_pcl and predictions["depth"][1] is not None:
                xyz_points = predictions["depth"][1].to(self.cpu_device)
                color_image = panoptic_output.get_image() if panoptic_output is not None else image
                pcl_output = visualizer.draw_pcl(xyz_points, color_image)

        return panoptic_output, depth_output, pcl_output

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object.

        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        video_visualizer = MGNetVideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictor):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            predictions = predictor(frame)
            panoptic_output = None
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                panoptic_output = video_visualizer.draw_panoptic_seg_predictions(
                    frame, panoptic_seg.to(self.cpu_device), segments_info
                )
            depth_output = None
            pcl_output = None
            if "depth" in predictions:
                depth_output = video_visualizer.draw_depth(
                    frame, predictions["depth"][0].to(self.cpu_device)
                )
                if self.show_pcl and predictions["depth"][1] is not None:
                    xyz_points = predictions["depth"][1].to(self.cpu_device)
                    color_image = (
                        panoptic_output.get_image() if panoptic_output is not None else frame
                    )
                    pcl_output = video_visualizer.draw_pcl(frame, xyz_points, color_image)

            return panoptic_output, depth_output, pcl_output

        frame_gen = self._frame_from_video(video)
        for frame in frame_gen:
            yield process_predictions(frame, self.predictor)

    @staticmethod
    def _frame_from_video(video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_mgnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Disable DGC scaling if no calibration_file is given
    if not args.calibration_file:
        cfg.MODEL.POST_PROCESSING.USE_DGC_SCALING = False
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="MGNet demo")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--calibration-file",
        help="Calibration used for DGC. If None is provided, depth is visualized unscaled.",
    )
    parser.add_argument(
        "--show-pcl", action="store_true", default=False, help="Show 3D pcl using Open3d."
    )
    parser.add_argument(
        "--pcl-view-point", help="Path to json cam param file used for open3d PCL visualization."
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def draw_or_save_visualizations(
    output_filename=None,
    visualized_panoptic=None,
    visualized_depth=None,
    ax1=None,
    ax2=None,
    visualized_pcl=None,
    vis=None,
    pcl_view_point=None,
    update_renderer=True,
):
    if visualized_pcl is not None and vis is not None:
        vis.clear_geometries()
        vis.add_geometry(visualized_pcl)
        if pcl_view_point is not None:
            ctr = vis.get_view_control()
            param = open3d.io.read_pinhole_camera_parameters(pcl_view_point)
            ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer() if update_renderer else vis.run()
        if not update_renderer:
            param = vis.get_view_control().convert_to_pinhole_camera_parameters()
            open3d.io.write_pinhole_camera_parameters("open3d_cam.json", param)
            vis.destroy_window()
        if output_filename is not None:
            vis.capture_screen_image(
                str("".join(output_filename.split(".")[:-1]))
                + "_pcl."
                + str(output_filename.split(".")[-1])
            )
    if output_filename is not None:
        if visualized_panoptic is not None:
            visualized_panoptic.save(
                str("".join(output_filename.split(".")[:-1]))
                + "_panoptic."
                + str(output_filename.split(".")[-1])
            )
        if visualized_depth is not None:
            visualized_depth.save(
                str("".join(output_filename.split(".")[:-1]))
                + "_depth."
                + str(output_filename.split(".")[-1])
            )
    else:
        if visualized_panoptic is not None:
            ax1.imshow(visualized_panoptic.get_image(), animated=True)
        if visualized_depth is not None:
            ax2.imshow(visualized_depth.get_image(), animated=True)
        plt.draw()
        if update_renderer:
            plt.pause(0.05)
        else:
            plt.show()
    return True


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_cityscapes_scene_seg(_root)
    register_all_kitti_eigen_scene_seg(_root)

    demo = VisualizationDemo(cfg, args.calibration_file, args.show_pcl)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set_title("Panoptic prediction overlay")
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_title("Depth prediction")

    vis = None
    pcl_view_point = args.pcl_view_point if args.pcl_view_point is not None else None
    if args.show_pcl:
        vis = open3d.visualization.Visualizer()
        vis.create_window(width=512, height=512)

    if args.input:
        if len(args.input) == 1:
            args.input = sorted(glob.glob(os.path.expanduser(args.input[0])))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="RGB")
            start_time = time.time()
            visualized_panoptic, visualized_depth, visualized_pcl = demo.run_on_image(img)
            output_filename = (
                os.path.join(args.output, os.path.basename(path)) if args.output else None
            )
            if not draw_or_save_visualizations(
                output_filename=output_filename,
                visualized_panoptic=visualized_panoptic,
                visualized_depth=visualized_depth,
                ax1=ax1,
                ax2=ax2,
                visualized_pcl=visualized_pcl,
                vis=vis,
                pcl_view_point=pcl_view_point,
                update_renderer=True if len(args.input) > 1 else False,
            ):
                break
    elif args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        frame = 0
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)
        assert os.path.isfile(args.video_input)
        for visualized_panoptic, visualized_depth, visualized_pcl in tqdm.tqdm(
            demo.run_on_video(video), total=num_frames
        ):
            output_filename = (
                os.path.join(args.output, "frame_" + str(frame).zfill(7) + ".png")
                if args.output
                else None
            )
            if not draw_or_save_visualizations(
                output_filename=output_filename,
                visualized_panoptic=visualized_panoptic,
                visualized_depth=visualized_depth,
                ax1=ax1,
                ax2=ax2,
                visualized_pcl=visualized_pcl,
                vis=vis,
                pcl_view_point=pcl_view_point,
            ):
                break
            frame += 1
        video.release()
        if not args.output:
            cv2.destroyAllWindows()
