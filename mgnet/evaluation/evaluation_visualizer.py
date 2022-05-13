import cv2
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator
from mgnet.inference.visualizer import MGNetVisualizer

__all__ = ["EvaluationVisualizer"]


class EvaluationVisualizer(DatasetEvaluator):
    """
    Visualizes prediction images during evaluation using matplotlib plots.
    Should be disabled during training and only used for debug purposes.
    """

    def __init__(self, dataset_name):
        self._metadata = MetadataCatalog.get(dataset_name)
        self.fig, self.ax = plt.subplots(3, 1)

    def process(self, inputs, outputs):
        for _input, output in zip(inputs, outputs):
            img = _input["image"].permute(1, 2, 0).detach().cpu().numpy()
            # Resize image back to original size in case it was  changed for model inference
            if img.shape[:2] != (_input["height"], _input["width"]):
                img = cv2.resize(img, (_input["width"], _input["height"]))
            self.ax[0].imshow(img)
            vis = MGNetVisualizer(img, metadata=self._metadata)
            if "panoptic_seg" in output:
                panoptic_seg, segments_info = output["panoptic_seg"]
                panoptic_prediction = vis.draw_panoptic_seg(
                    panoptic_seg.detach().cpu(), segments_info
                )
                self.ax[1].imshow(panoptic_prediction.get_image())
            if "depth" in output:
                depth_prediction = vis.draw_depth(output["depth"][0].detach().cpu())
                self.ax[2].imshow(depth_prediction.get_image())
            plt.draw()
            plt.pause(0.5)
