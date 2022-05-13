import json
import os
from typing import Dict, List

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager

from .cityscapes_scene_seg import CITYSCAPES_CATEGORIES, CITYSCAPES_SCENE_SEG_CATEGORIES

__all__ = ["register_all_kitti_eigen_scene_seg"]

IMAGE_FOLDER = {
    "left": "image_02",
    "right": "image_03",
}

# Name of different calibration files
CALIB_FILE = {
    "cam2cam": "calib_cam_to_cam.txt",
    "velo2cam": "calib_velo_to_cam.txt",
    "imu2velo": "calib_imu_to_velo.txt",
}

_RAW_KITTI_EIGEN_SCENE_SEG_SPLITS = {
    "kitti_zhou_scene_seg_train": (
        "kitti_eigen/data_splits/eigen_zhou_files.txt",
        "kitti_eigen/panoptic_pseudo_labels/eigen_zhou_files_panoptic",
        "kitti_eigen/panoptic_pseudo_labels/eigen_zhou_files_panoptic.json",
    ),
    "kitti_eigen_scene_seg_test": (
        "kitti_eigen/data_splits/eigen_test_files.txt",
        "kitti_eigen/panoptic_pseudo_labels/eigen_test_files_panoptic",
        "kitti_eigen/panoptic_pseudo_labels/eigen_test_files_panoptic.json",
    ),
}


def load_kitti_eigen_scene_seg(
    root: str,
    image_split_file: str,
    gt_dir: str,
    gt_json: str,
    meta: Dict,
    pseudo_label_generation: bool,
) -> List[dict]:
    """
    Args:
       root (str): path to dataset root.
       image_split_file (str): path to the data split file.
       gt_dir (str): path to the raw annotations.
           e.g., "~/kitti_eigen/gtFine_sequence/kitti_eigen_panoptic_train".
       gt_json (str): path to the json file.
           e.g., "~/kitti_eigen/gtFine_sequence/kitti_eigen_panoptic_train.json".
       meta (dict): dictionary containing "thing_dataset_id_to_contiguous_id" and
           "stuff_dataset_id_to_contiguous_id" to map category ids to contiguous ids for training.
       pseudo_label_generation (bool): If true, skip assertions and only load image files.

    Returns:
       list[dict]: a list of dicts in Detectron2 standard format. (See
           `Using Custom Datasets </tutorials/datasets.html>`_ )
    """

    def _convert_category_id(segment_info, meta):
        if segment_info["category_id"] in meta["thing_dataset_id_to_contiguous_id"]:
            segment_info["category_id"] = meta["thing_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        else:
            segment_info["category_id"] = meta["stuff_dataset_id_to_contiguous_id"][
                segment_info["category_id"]
            ]
        return segment_info

    calibration_cache = {}

    with open(image_split_file) as f:
        files = f.read().splitlines()
    files = [(x.split(" ")[0], "", "") for x in files]

    if not pseudo_label_generation and ("train" in gt_dir or "zhou" in gt_dir):
        assert os.path.exists(
            gt_json
        ), "Please run `python tools/generate_pseudo_labels.py` to generate pseudo label files."
        with open(gt_json) as f:
            json_info = json.load(f)
        files = _get_panoptic_files(files, gt_dir, json_info)
    ret = []
    for file, label_file, segments_info in files:
        image_file = os.path.join(root, "kitti_eigen", file)

        # Get previous and next frame for current image_file from the video sequence dir
        image_idx = int(image_file.split("/")[-1][:-4])
        image_prev_file = image_file[:-14] + str(image_idx - 1).zfill(10) + image_file[-4:]
        image_next_file = image_file[:-14] + str(image_idx + 1).zfill(10) + image_file[-4:]

        # Skip first and last samples in video sequence in train set
        if ("train" in gt_dir or "zhou" in gt_dir) and (
            not os.path.exists(image_prev_file) or not os.path.exists(image_next_file)
        ):
            continue

        depth_file = _get_depth_file(image_file)

        # Skip test sample if depth file is not available
        if "test" in gt_dir and not os.path.exists(depth_file):
            continue

        # Add intrinsics
        parent_folder = _get_parent_folder(image_file)
        if parent_folder in calibration_cache:
            c_data = calibration_cache[parent_folder]
        else:
            c_data = _read_raw_calib_file(parent_folder)
            calibration_cache[parent_folder] = c_data
        intrinsics = _get_intrinsics(image_file, c_data)

        # Convert to Cityscapes format
        calibration_info = dict()
        calibration_info["intrinsic"] = dict()
        calibration_info["intrinsic"]["fx"] = intrinsics[0][0]
        calibration_info["intrinsic"]["fy"] = intrinsics[1][1]
        calibration_info["intrinsic"]["u0"] = intrinsics[0][2]
        calibration_info["intrinsic"]["v0"] = intrinsics[1][2]
        calibration_info["extrinsic"] = dict()
        calibration_info["extrinsic"]["baseline"] = 0.54
        calibration_info["extrinsic"]["z"] = 1.65

        segments_info = [_convert_category_id(x, meta) for x in segments_info]
        ret.append(
            {
                "file_name": image_file,
                "image_id": file[:-4],
                "pan_seg_file_name": label_file,
                "depth_file_name": depth_file,
                "prev_img_file_name": image_prev_file,
                "next_img_file_name": image_next_file,
                "segments_info": segments_info,
                "calibration_info": calibration_info,
            }
        )
    assert len(ret), f"No images found from data split file {image_split_file}!"
    if not pseudo_label_generation and ("train" in gt_dir or "zhou" in gt_dir):
        assert PathManager.isfile(
            ret[0]["pan_seg_file_name"]
        ), "Please generate panoptic annotation with python datasets/prepare_kitti_eigen.py"
    return ret


def register_all_kitti_eigen_scene_seg(root, pseudo_label_generation=False):
    meta = {}
    # Use scene seg categories for pseudo label generation, since a scene seg model is used
    categories = (
        CITYSCAPES_SCENE_SEG_CATEGORIES if pseudo_label_generation else CITYSCAPES_CATEGORIES
    )
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in categories]
    thing_colors = [k["color"] for k in categories]
    stuff_classes = [k["name"] for k in categories]
    stuff_colors = [k["color"] for k in categories]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    meta["categories"] = categories
    ignore_in_eval = []
    for k in categories:
        if k["ignoreInEval"]:
            ignore_in_eval.append({"id": k["id"], "trainId": k["trainId"]})
    meta["ignore_in_eval"] = ignore_in_eval

    # There are three types of ids in cityscapes panoptic segmentation:
    # (1) category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the classifier
    # (2) instance id: this id is used to differentiate different instances from
    #   the same category. For "stuff" classes, the instance id is always 0; for
    #   "thing" classes, the instance id starts from 1 and 0 is reserved for
    #   ignored instances (e.g. crowd annotation).
    # (3) panoptic id: this is the compact id that encode both category and
    #   instance id by: category_id * 1000 + instance_id.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for k in categories:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (
        image_file_list,
        gt_dir,
        gt_json,
    ) in _RAW_KITTI_EIGEN_SCENE_SEG_SPLITS.items():

        image_file_list = os.path.join(root, image_file_list)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        # fmt: off
        DatasetCatalog.register(
            key,
            lambda w=root, x=image_file_list, y=gt_dir, z=gt_json:
            load_kitti_eigen_scene_seg(
                w, x, y, z, meta, pseudo_label_generation
            ),
        )
        # fmt: on

        MetadataCatalog.get(key).set(
            image_file_list=image_file_list,
            panoptic_root=gt_dir,
            panoptic_json=gt_json,
            gt_dir="/".join(gt_dir.split("/")[:-1]),
            evaluator_type="kitti_eigen_scene_seg",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )


def _get_parent_folder(image_file):
    """Get the parent folder from image_file."""
    return os.path.abspath(os.path.join(image_file, "../../../.."))


def _get_panoptic_files(list_files, gt_dir, json_info):
    """Get panoptic annotations and image files from json."""
    files = []
    for ann in json_info["annotations"]:
        image_file = ann["file_name"].replace("label_", "image_")
        label_file = os.path.join(gt_dir, ann["file_name"])
        segments_info = ann["segments_info"]

        files.append((image_file, label_file, segments_info))

    assert len(files), "No images found"
    assert len(list_files) == len(files), "Not all annotations dor image list were found!"
    assert PathManager.isfile(files[0][1]), files[0][1]
    return files


def _get_depth_file(image_file):
    """Get the corresponding depth file from an image file."""
    for cam in ["left", "right"]:
        if IMAGE_FOLDER[cam] in image_file:
            depth_file = image_file.replace(
                IMAGE_FOLDER[cam] + "/data",
                "proj_depth/{}/{}".format("groundtruth", IMAGE_FOLDER[cam]),
            )
            return depth_file


def _get_intrinsics(image_file, calib_data):
    """Get intrinsics from the calib_data dictionary."""
    for cam in ["left", "right"]:
        # Check for both cameras, if found replace and return intrinsics
        if IMAGE_FOLDER[cam] in image_file:
            return np.reshape(calib_data[IMAGE_FOLDER[cam].replace("image", "P_rect")], (3, 4))[
                :, :3
            ]


def _read_raw_calib_file(folder):
    """Read raw calibration files from folder."""
    filepath = os.path.join(folder, CALIB_FILE["cam2cam"])

    data = {}
    with open(filepath, "r") as f:
        for line in f.readlines():
            key, value = line.split(":", 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data
