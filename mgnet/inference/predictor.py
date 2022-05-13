import json

import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import transforms as T
from detectron2.modeling import build_model

__all__ = ["MGNetPredictor"]


class MGNetPredictor:
    """
    MGNetPredictor to run inference on single images.
    A calibration_file can be provided for metric depth prediction using DGC scaling.
    """

    def __init__(self, cfg, calibration_file=None):
        """
        Args:
            cfg: detectron2 CfgNode with model config
            calibration_file: path to calibration json file in Cityscapes format
        """
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format
        self.amp_enabled = cfg.TEST.AMP.ENABLED

        self.calibration_info = None
        if calibration_file is not None:
            with open(calibration_file, "r") as f:
                self.calibration_info = json.load(f)

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict): the output of the model for one image only.
        """
        with torch.no_grad():
            # Apply pre-processing to image.
            if self.input_format == "BGR":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.transpose(2, 0, 1).copy())

            input_dict = {
                "image": image,
                "height": height,
                "width": width,
            }

            if self.calibration_info is not None:
                fx = self.calibration_info["intrinsic"]["fx"]
                fy = self.calibration_info["intrinsic"]["fy"]
                u0 = self.calibration_info["intrinsic"]["u0"]
                v0 = self.calibration_info["intrinsic"]["v0"]
                camera_matrix = torch.tensor(
                    [[fx, 0, u0], [0, fy, v0], [0, 0, 1]], dtype=torch.float32
                )

                input_dict.update(
                    {
                        "camera_matrix": camera_matrix,
                        "camera_height": torch.tensor(self.calibration_info["extrinsic"]["z"]),
                    }
                )

            with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                predictions = self.model([input_dict])[0]
            return predictions
