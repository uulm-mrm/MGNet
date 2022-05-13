import contextlib
import io
import itertools
import json
import os
import tempfile
from collections import OrderedDict
from typing import Optional

import numpy as np
from detectron2.evaluation.cityscapes_evaluation import CityscapesEvaluator
from detectron2.utils import comm
from detectron2.utils.file_io import PathManager
from PIL import Image
from tabulate import tabulate

__all__ = ["PanopticEvaluator"]


class PanopticEvaluator(CityscapesEvaluator):
    """
    Evaluate Panoptic Quality metrics.
    It saves panoptic segmentation prediction in `output_dir`

    It contains a synchronize call and has to be called from all workers.

    Based on the detectron2 COCOPanopticEvaluator
    (See https://github.com/facebookresearch/detectron2/blob/main/detectron2/evaluation/panoptic_evaluation.py),  # noqa
    but uses ignore_in_eval categories from the metadata to ignore categories during eval, i.e. ego_car.
    """

    def __init__(self, dataset_name: str, output_dir: Optional[str] = None):
        """
        Args:
            dataset_name: name of the dataset
            output_dir: output directory to save results for evaluation.
        """
        super().__init__(dataset_name)
        self._thing_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
        }
        self._stuff_contiguous_id_to_dataset_id = {
            v: k for k, v in self._metadata.stuff_dataset_id_to_contiguous_id.items()
        }
        self._ignored_cats = self._metadata.ignore_in_eval

        self._output_dir = output_dir
        if self._output_dir is not None:
            PathManager.mkdirs(self._output_dir)

    def reset(self):
        self._predictions = []

    def _convert_category_id(self, segment_info):
        isthing = segment_info.pop("isthing", None)
        if isthing is None:
            # the model produces panoptic category id directly. No more conversion needed
            return segment_info
        if isthing is True:
            segment_info["category_id"] = self._thing_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = self._stuff_contiguous_id_to_dataset_id[
                segment_info["category_id"]
            ]
        return segment_info

    def process(self, inputs, outputs):
        from panopticapi.utils import id2rgb

        for _input, output in zip(inputs, outputs):
            panoptic_img, segments_info = output["panoptic_seg"]
            panoptic_img = panoptic_img.cpu().numpy()
            # Assign ignored ids to VOID region.
            label_divisor = self._metadata.label_divisor
            for cat in self._ignored_cats:
                panoptic_img[panoptic_img // label_divisor == cat["trainId"]] = -1
            if segments_info is None:
                # If "segments_info" is None, we assume "panoptic_img" is a
                # H*W int32 image storing the panoptic_id in the format of
                # category_id * label_divisor + instance_id. We reserve -1 for
                # VOID label, and add 1 to panoptic_img since the official
                # evaluation script uses 0 for VOID label.
                segments_info = []
                for panoptic_label in np.unique(panoptic_img):
                    if panoptic_label == -1:
                        # VOID region.
                        continue
                    pred_class = panoptic_label // label_divisor
                    isthing = (
                        pred_class in self._metadata.thing_dataset_id_to_contiguous_id.values()
                    )
                    segments_info.append(
                        {
                            "id": int(panoptic_label) + 1,
                            "category_id": int(pred_class),
                            "isthing": bool(isthing),
                        }
                    )
                # Official evaluation script uses 0 for VOID label.
                panoptic_img += 1

            file_name = os.path.basename(_input["file_name"])
            file_name_png = os.path.splitext(file_name)[0] + ".png"
            with io.BytesIO() as out:
                Image.fromarray(id2rgb(panoptic_img)).save(out, format="PNG")
                segments_info = [self._convert_category_id(x) for x in segments_info]
                self._predictions.append(
                    {
                        "image_id": _input["image_id"],
                        "file_name": file_name_png,
                        "png_string": out.getvalue(),
                        "segments_info": segments_info,
                    }
                )

    def evaluate(self):
        comm.synchronize()

        self._predictions = comm.gather(self._predictions)
        self._predictions = list(itertools.chain(*self._predictions))
        if not comm.is_main_process():
            return

        # PanopticApi requires local files
        gt_json = PathManager.get_local_path(self._metadata.panoptic_json)
        gt_folder = PathManager.get_local_path(self._metadata.panoptic_root)

        with tempfile.TemporaryDirectory(prefix="panoptic_eval") as pred_dir:
            self._logger.info("Writing all panoptic predictions to {} ...".format(pred_dir))
            for p in self._predictions:
                with open(os.path.join(pred_dir, p["file_name"]), "wb") as f:
                    f.write(p.pop("png_string"))

            with open(gt_json, "r") as f:
                json_data = json.load(f)

            # Assign ignored ids in GT to VOID region.
            ignored_dataset_ids = [cat["id"] for cat in self._ignored_cats]
            filtered_categories = []
            for cat in json_data["categories"]:
                if cat["id"] not in ignored_dataset_ids:
                    filtered_categories.append(cat)
            json_data["categories"] = filtered_categories

            output_dir = self._output_dir or pred_dir
            filtered_gt_json = os.path.join(output_dir, "filtered_gt.json")
            with PathManager.open(filtered_gt_json, "w") as f:
                f.write(json.dumps(json_data))

            json_data["annotations"] = self._predictions
            predictions_json = os.path.join(output_dir, "predictions.json")
            with PathManager.open(predictions_json, "w") as f:
                f.write(json.dumps(json_data))

            from panopticapi.evaluation import pq_compute

            with contextlib.redirect_stdout(io.StringIO()):
                pq_res = pq_compute(
                    PathManager.get_local_path(filtered_gt_json),
                    PathManager.get_local_path(predictions_json),
                    gt_folder=gt_folder,
                    pred_folder=pred_dir,
                )

        res = {}
        res["PQ"] = 100 * pq_res["All"]["pq"]
        res["SQ"] = 100 * pq_res["All"]["sq"]
        res["RQ"] = 100 * pq_res["All"]["rq"]
        res["PQ_th"] = 100 * pq_res["Things"]["pq"]
        res["SQ_th"] = 100 * pq_res["Things"]["sq"]
        res["RQ_th"] = 100 * pq_res["Things"]["rq"]
        res["PQ_st"] = 100 * pq_res["Stuff"]["pq"]
        res["SQ_st"] = 100 * pq_res["Stuff"]["sq"]
        res["RQ_st"] = 100 * pq_res["Stuff"]["rq"]

        results = OrderedDict({"panoptic_seg": res})
        self._print_panoptic_results(pq_res)

        return results

    def _print_panoptic_results(self, pq_res):
        headers = ["", "PQ", "SQ", "RQ", "#categories"]
        data = []
        for name in ["All", "Things", "Stuff"]:
            row = [name] + [pq_res[name][k] * 100 for k in ["pq", "sq", "rq"]] + [pq_res[name]["n"]]
            data.append(row)
        table = tabulate(
            data,
            headers=headers,
            tablefmt="pipe",
            floatfmt=".3f",
            stralign="center",
            numalign="center",
        )
        self._logger.info("Panoptic Evaluation Results:\n" + table)
