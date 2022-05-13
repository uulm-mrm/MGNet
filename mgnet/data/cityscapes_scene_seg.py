import copy
import json
import os
from typing import Dict, List

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.cityscapes_panoptic import get_cityscapes_panoptic_files
from detectron2.utils.file_io import PathManager

__all__ = [
    "CITYSCAPES_CATEGORIES",
    "CITYSCAPES_SCENE_SEG_CATEGORIES",
    "register_all_cityscapes_scene_seg",
]

# fmt: off
CITYSCAPES_CATEGORIES = [
    {"color": (128,  64, 128), "isthing": 0, "id":  7, "trainId":  0, "ignoreInEval": False, "name":          "road"},  # noqa
    {"color": (244,  35, 232), "isthing": 0, "id":  8, "trainId":  1, "ignoreInEval": False, "name":      "sidewalk"},  # noqa
    {"color": ( 70,  70,  70), "isthing": 0, "id": 11, "trainId":  2, "ignoreInEval": False, "name":      "building"},  # noqa
    {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId":  3, "ignoreInEval": False, "name":          "wall"},  # noqa
    {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId":  4, "ignoreInEval": False, "name":         "fence"},  # noqa
    {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId":  5, "ignoreInEval": False, "name":          "pole"},  # noqa
    {"color": (250, 170,  30), "isthing": 0, "id": 19, "trainId":  6, "ignoreInEval": False, "name": "traffic light"},  # noqa
    {"color": (220, 220,   0), "isthing": 0, "id": 20, "trainId":  7, "ignoreInEval": False, "name":  "traffic sign"},  # noqa
    {"color": (107, 142,  35), "isthing": 0, "id": 21, "trainId":  8, "ignoreInEval": False, "name":    "vegetation"},  # noqa
    {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId":  9, "ignoreInEval": False, "name":       "terrain"},  # noqa
    {"color": ( 70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "ignoreInEval": False, "name":           "sky"},  # noqa
    {"color": (220,  20,  60), "isthing": 1, "id": 24, "trainId": 11, "ignoreInEval": False, "name":        "person"},  # noqa
    {"color": (255,   0,   0), "isthing": 1, "id": 25, "trainId": 12, "ignoreInEval": False, "name":         "rider"},  # noqa
    {"color": (  0,   0, 142), "isthing": 1, "id": 26, "trainId": 13, "ignoreInEval": False, "name":           "car"},  # noqa
    {"color": (  0,   0,  70), "isthing": 1, "id": 27, "trainId": 14, "ignoreInEval": False, "name":         "truck"},  # noqa
    {"color": (  0,  60, 100), "isthing": 1, "id": 28, "trainId": 15, "ignoreInEval": False, "name":           "bus"},  # noqa
    {"color": (  0,  80, 100), "isthing": 1, "id": 31, "trainId": 16, "ignoreInEval": False, "name":         "train"},  # noqa
    {"color": (  0,   0, 230), "isthing": 1, "id": 32, "trainId": 17, "ignoreInEval": False, "name":    "motorcycle"},  # noqa
    {"color": (119,  11,  32), "isthing": 1, "id": 33, "trainId": 18, "ignoreInEval": False, "name":       "bicycle"},  # noqa
]

# Add ego vehicle category for scene seg
CITYSCAPES_SCENE_SEG_CATEGORIES = [
    {"color": ( 72, 209, 204), "isthing": 0, "id":  1, "trainId":  0, "ignoreInEval":  True, "name":   "ego vehicle"},  # noqa
]
# fmt: on

for cat in copy.deepcopy(CITYSCAPES_CATEGORIES):
    cat["trainId"] += 1
    CITYSCAPES_SCENE_SEG_CATEGORIES.append(cat)

_RAW_CITYSCAPES_SCENE_SEG_SPLITS = {
    "cityscapes_fine_scene_seg_train": (
        "cityscapes/leftImg8bit/train",
        "cityscapes/leftImg8bit_sequence/train",
        "cityscapes/camera/train",
        "cityscapes/disparity/train",
        "cityscapes/gtFine/cityscapes_panoptic_train",
        "cityscapes/gtFine/cityscapes_panoptic_train.json",
    ),
    "cityscapes_scene_seg_train_video_sequence": (
        "cityscapes/leftImg8bit_sequence/train",
        "cityscapes/leftImg8bit_sequence/train",
        "cityscapes/camera/train",
        "cityscapes/disparity/train",
        "cityscapes/gtFine_sequence/cityscapes_panoptic_train",
        "cityscapes/gtFine_sequence/cityscapes_panoptic_train.json",
    ),
    "cityscapes_fine_scene_seg_val": (
        "cityscapes/leftImg8bit/val",
        "cityscapes/leftImg8bit_sequence/val",
        "cityscapes/camera/val",
        "cityscapes/disparity/val",
        "cityscapes/gtFine/cityscapes_panoptic_val",
        "cityscapes/gtFine/cityscapes_panoptic_val.json",
    ),
    # "cityscapes_fine_scene_seg_test": not supported yet
}


def load_cityscapes_scene_seg(
    image_dir: str,
    image_seq_dir: str,
    camera_dir: str,
    disparity_dir: str,
    gt_dir: str,
    gt_json: str,
    meta: Dict,
    pseudo_label_generation: bool,
) -> List[dict]:
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        image_seq_dir (str): path to the raw video sequences.
            e.g., "~/cityscapes/leftImg8bit_sequence/train".
        camera_dir (str): path to camera calibration files. e.g., "~/cityscapes/camera/train".
        disparity_dir (str): path to disparity images. e.g., "~/cityscapes/disparity/train".
        gt_dir (str): path to the raw annotations.
            e.g., "~/cityscapes/gtFine/cityscapes_panoptic_train".
        gt_json (str): path to the json file.
            e.g., "~/cityscapes/gtFine/cityscapes_panoptic_train.json".
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

    if pseudo_label_generation:
        # Load image files only
        files = []
        for path, subdirs, file_list in os.walk(image_dir):
            for name in file_list:
                files.append((os.path.join(path, name), "", ""))
    else:
        # Load image files with annotations
        assert os.path.exists(
            gt_json
        ), "Please run `python prepare_cityscapes.py` to generate label files."
        with open(gt_json) as f:
            json_info = json.load(f)
        files = get_cityscapes_panoptic_files(image_dir, gt_dir, json_info)
    ret = []
    for image_file, label_file, segments_info in files:
        sem_label_file = (
            image_file.replace("leftImg8bit", "gtFine").split(".")[0] + "_labelTrainIds.png"
        )

        # Get previous and next frame for current image_file from the video sequence dir
        image_sequence_file = os.path.join(image_seq_dir, "/".join(image_file.split("/")[-2:]))
        image_idx = int(image_sequence_file.split("/")[-1][-22:-16])
        image_prev_file = (
            image_sequence_file[0:-22] + str(image_idx - 1).zfill(6) + image_sequence_file[-16:]
        )
        image_next_file = (
            image_sequence_file[0:-22] + str(image_idx + 1).zfill(6) + image_sequence_file[-16:]
        )

        # Skip first and last samples in video sequence in train set
        if "train" in gt_dir and (
            not os.path.exists(image_prev_file) or not os.path.exists(image_next_file)
        ):
            continue

        disparity_file = os.path.join(disparity_dir, "/".join(image_file.split("/")[-2:])).replace(
            "_leftImg8bit.png", "_disparity.png"
        )

        camera_info_file = os.path.join(camera_dir, "/".join(image_file.split("/")[-2:])).replace(
            "_leftImg8bit.png", "_camera.json"
        )

        # The camera info file is not available for all sequence images. However, the camera info
        # is constant within one drive, so we list all info files in the current drive dir and
        # select the first one
        camera_info_path = "/".join(camera_info_file.split("/")[:-1])
        camera_info_file_list = os.listdir(camera_info_path)
        with open(os.path.join(camera_info_path, camera_info_file_list[0]), "r") as f:
            calibration_info = json.load(f)

        segments_info = [_convert_category_id(x, meta) for x in segments_info]
        ret.append(
            {
                "file_name": image_file,
                "image_id": "_".join(
                    os.path.splitext(os.path.basename(image_file))[0].split("_")[:3]
                ),
                "sem_seg_file_name": sem_label_file,
                "pan_seg_file_name": label_file,
                "disparity_file_name": disparity_file,
                "prev_img_file_name": image_prev_file,
                "next_img_file_name": image_next_file,
                "segments_info": segments_info,
                "calibration_info": calibration_info,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    if not pseudo_label_generation:
        assert PathManager.isfile(
            ret[0]["pan_seg_file_name"]
        ), "Please generate panoptic annotation with python cityscapesscripts/preparation/createPanopticImgs.py"  # noqa
    return ret


def register_all_cityscapes_scene_seg(root, pseudo_label_generation=False):
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    meta["thing_classes"] = [k["name"] for k in CITYSCAPES_SCENE_SEG_CATEGORIES]
    meta["thing_colors"] = [k["color"] for k in CITYSCAPES_SCENE_SEG_CATEGORIES]
    meta["stuff_classes"] = [k["name"] for k in CITYSCAPES_SCENE_SEG_CATEGORIES]
    meta["stuff_colors"] = [k["color"] for k in CITYSCAPES_SCENE_SEG_CATEGORIES]

    meta["categories"] = CITYSCAPES_SCENE_SEG_CATEGORIES
    ignore_in_eval = []
    for k in CITYSCAPES_SCENE_SEG_CATEGORIES:
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

    for k in CITYSCAPES_SCENE_SEG_CATEGORIES:
        if k["isthing"] == 1:
            thing_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]
        else:
            stuff_dataset_id_to_contiguous_id[k["id"]] = k["trainId"]

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    for key, (
        image_dir,
        image_seq_dir,
        camera_dir,
        disparity_dir,
        gt_dir,
        gt_json,
    ) in _RAW_CITYSCAPES_SCENE_SEG_SPLITS.items():
        image_dir = os.path.join(root, image_dir)
        image_seq_dir = os.path.join(root, image_seq_dir)
        camera_dir = os.path.join(root, camera_dir)
        disparity_dir = os.path.join(root, disparity_dir)
        gt_dir = os.path.join(root, gt_dir)
        gt_json = os.path.join(root, gt_json)

        # fmt: off
        DatasetCatalog.register(
            key,
            lambda u=image_dir, v=image_seq_dir, w=camera_dir, x=disparity_dir, y=gt_dir, z=gt_json:
                load_cityscapes_scene_seg(
                    u, v, w, x, y, z, meta, pseudo_label_generation
                ),
        )
        # fmt: on
        MetadataCatalog.get(key).set(
            panoptic_root=gt_dir,
            image_root=image_dir,
            panoptic_json=gt_json,
            gt_dir=gt_dir.replace("cityscapes_panoptic_", ""),
            evaluator_type="cityscapes_scene_seg",
            ignore_label=255,
            label_divisor=1000,
            **meta,
        )
