import itertools

import cv2
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm
from mgnet.inference.visualizer import MGNetVisualizer
from torch.utils.tensorboard import SummaryWriter

__all__ = ["TensorboardImageWriter"]


class TensorboardImageWriter(DatasetEvaluator):
    """
    Writes prediction images to Tensorboard during evaluations runs.
    Limited to two images per worker to keep time and memory footprint small.
    """

    _step = 0

    def __init__(self, dataset_name, log_dir, eval_period=None):
        self._metadata = MetadataCatalog.get(dataset_name)
        self._panoptic_prediction_list = []
        self._depth_prediction_list = []
        self._writer = SummaryWriter(log_dir)
        self._eval_period = 1 if eval_period is None else eval_period

    def reset(self):
        self._panoptic_prediction_list = []
        self._depth_prediction_list = []
        TensorboardImageWriter._step += self._eval_period

    def process(self, inputs, outputs):
        # We only monitor 2 predictions per worker to keep the computation low
        if len(self._panoptic_prediction_list) == 2 or len(self._depth_prediction_list) == 2:
            return
        for _input, output in zip(inputs, outputs):
            img = _input["image"].permute(1, 2, 0).detach().cpu().numpy()
            # Resize image back to original size in case it was  changed for model inference
            if img.shape[:2] != (_input["height"], _input["width"]):
                img = cv2.resize(img, (_input["width"], _input["height"]))
            vis = MGNetVisualizer(img, metadata=self._metadata)
            if "panoptic_seg" in output:
                panoptic_seg, segments_info = output["panoptic_seg"]
                panoptic_prediction = vis.draw_panoptic_seg(
                    panoptic_seg.detach().cpu(), segments_info
                )
                self._panoptic_prediction_list.append(panoptic_prediction.get_image())
            if "depth" in output:
                depth_prediction = vis.draw_depth(output["depth"][0].detach().cpu())
                self._depth_prediction_list.append(depth_prediction.get_image())
            break

    def evaluate(self):
        comm.synchronize()

        # Gather predictions from all workers
        self._panoptic_prediction_list = comm.gather(self._panoptic_prediction_list)
        self._panoptic_prediction_list = list(itertools.chain(*self._panoptic_prediction_list))
        self._depth_prediction_list = comm.gather(self._depth_prediction_list)
        self._depth_prediction_list = list(itertools.chain(*self._depth_prediction_list))

        if not comm.is_main_process():
            return

        # Add images to tensorboard writer
        if len(self._panoptic_prediction_list) > 0:
            self._writer.add_images(
                "Panoptic predictions",
                np.stack(self._pad_images_in_list(self._panoptic_prediction_list)),
                global_step=TensorboardImageWriter._step,
                dataformats="NHWC",
            )
        if len(self._depth_prediction_list) > 0:
            self._writer.add_images(
                "Depth predictions",
                np.stack(self._pad_images_in_list(self._depth_prediction_list)),
                global_step=TensorboardImageWriter._step,
                dataformats="NHWC",
            )

        self._writer.flush()

    @staticmethod
    def _pad_images_in_list(image_list):
        padded_image_list = []

        max_h, max_w = 0, 0
        for idx, val in enumerate(image_list):
            max_h = max(max_h, val.shape[0])
            max_w = max(max_w, val.shape[1])
        for idx, val in enumerate(image_list):
            padded_image_list.append(
                cv2.copyMakeBorder(
                    val,
                    0,
                    max_h - val.shape[0],
                    0,
                    max_w - val.shape[1],
                    cv2.BORDER_CONSTANT,
                    (0, 0, 0),
                )
            )
        return padded_image_list
