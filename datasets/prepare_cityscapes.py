#!/usr/bin/python3
#
# Converts the *instanceIds.png annotations of the Cityscapes dataset
# to COCO-style panoptic segmentation format (http://cocodataset.org/#format-data).
# The convertion is working for 'fine' set of the annotations.
#
# By default with this tool uses IDs specified in labels_cityscapes.py. You can use flag
# --use-train-id to get train ids for categories. 'ignoreInEval' categories are
# removed during the conversion.
#
# In panoptic segmentation format image_id is used to match predictions and ground truth.
# For cityscapes image_id has form <city>_123456_123456 and corresponds to the prefix
# of cityscapes image files.
#
# See https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createPanopticImgs.py  # noqa
#
# We extended the script to use multiple CPU cores for faster conversion.

import argparse
import glob
import json
import os
import sys
from functools import partial
from multiprocessing.dummy import Pool

import numpy as np
from datasets.labels_cityscapes import id2label, labels

# Image processing
from PIL import Image

# The main method
from tqdm import tqdm

__all__ = ["convert2panoptic"]


def convert2panoptic(cityscapes_path=None, output_folder=None, use_train_id=False, set_names=None):
    # Where to look for Cityscapes
    if set_names is None:
        set_names = ["val", "train", "test"]
    if cityscapes_path is None:
        if "CITYSCAPES_DATASET" in os.environ:
            cityscapes_path = os.environ["CITYSCAPES_DATASET"]
        else:
            cityscapes_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
        cityscapes_path = os.path.join(cityscapes_path, "gtFine")

    if output_folder is None:
        output_folder = cityscapes_path

    categories = []
    for label in labels:
        if label.ignoreInEval:
            continue
        categories.append(
            {
                "id": int(label.trainId) if use_train_id else int(label.id),
                "name": label.name,
                "color": label.color,
                "supercategory": label.category,
                "isthing": 1 if label.hasInstances else 0,
            }
        )

    for setName in set_names:
        # how to search for all ground truth
        search_fine = os.path.join(cityscapes_path, setName, "*", "*_instanceIds.png")
        # search files
        files_fine = glob.glob(search_fine)
        files_fine.sort()

        files = files_fine
        # quit if we did not find anything
        if not files:
            print(
                "Error: Did not find any files for {} set using matching pattern {}. "
                "Please consult the README.".format(setName, search_fine)
            )
            sys.exit(-1)
        # a bit verbose
        print("Converting {} annotation files for {} set.".format(len(files), setName))

        train_if_suffix = "_trainId" if use_train_id else ""
        output_base_file = "cityscapes_panoptic_{}{}".format(setName, train_if_suffix)
        out_file = os.path.join(output_folder, "{}.json".format(output_base_file))
        print(
            "Json file with the annotations in panoptic format will be saved in {}".format(out_file)
        )
        panoptic_folder = os.path.join(output_folder, output_base_file)
        if not os.path.isdir(panoptic_folder):
            print("Creating folder {} for panoptic segmentation PNGs".format(panoptic_folder))
            os.mkdir(panoptic_folder)
        print(
            "Corresponding segmentations in .png format will be saved in {}".format(panoptic_folder)
        )

        pool = Pool()
        global pbar
        pbar = tqdm(total=len(files), desc=f"Convert {setName} set to COCO format", file=sys.stdout)
        convert_single_file = partial(
            _convert_single_file, use_train_id=use_train_id, panoptic_folder=panoptic_folder
        )
        image_annotations = pool.map(convert_single_file, files)
        pool.close()
        pool.join()

        # combine each image_annotation into global lists
        images = []
        annotations = []
        for image_annotation in image_annotations:
            images.append(image_annotation["image"])
            annotations.append(image_annotation["annotation"])

        print("\nSaving the json file {}".format(out_file))
        d = {"images": images, "annotations": annotations, "categories": categories}
        with open(out_file, "w") as f:
            json.dump(d, f, sort_keys=True, indent=4)


def _convert_single_file(file, use_train_id, panoptic_folder):
    original_format = np.array(Image.open(file))

    file_name = os.path.basename(file)
    image_id = file_name.replace("_gtFine_instanceIds.png", "")
    input_file_name = file_name.replace("_instanceIds.png", "_leftImg8bit.png")
    out_file_name = file_name.replace("_instanceIds.png", "_panoptic.png")
    # image entry, id for image is its filename without extension
    image = {
        "id": image_id,
        "width": int(original_format.shape[1]),
        "height": int(original_format.shape[0]),
        "file_name": input_file_name,
    }

    pan_format = np.zeros((original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8)

    segment_ids = np.unique(original_format)
    segment_info = []
    for segment_id in segment_ids:
        if segment_id < 1000:
            semantic_id = segment_id
            is_crowd = 1
        else:
            semantic_id = segment_id // 1000
            is_crowd = 0
        label_info = id2label[semantic_id]
        category_id = label_info.trainId if use_train_id else label_info.id
        if label_info.ignoreInEval:
            continue
        if not label_info.hasInstances:
            is_crowd = 0

        mask = original_format == segment_id
        color = [segment_id % 256, segment_id // 256, segment_id // 256 // 256]
        pan_format[mask] = color

        area = np.sum(mask)  # segment area computation

        # bbox computation for a segment
        hor = np.sum(mask, axis=0)
        hor_idx = np.nonzero(hor)[0]
        x = hor_idx[0]
        width = hor_idx[-1] - x + 1
        vert = np.sum(mask, axis=1)
        vert_idx = np.nonzero(vert)[0]
        y = vert_idx[0]
        height = vert_idx[-1] - y + 1
        bbox = [int(x), int(y), int(width), int(height)]

        segment_info.append(
            {
                "id": int(segment_id),
                "category_id": int(category_id),
                "area": int(area),
                "bbox": bbox,
                "iscrowd": is_crowd,
            }
        )

    annotation = {"image_id": image_id, "file_name": out_file_name, "segments_info": segment_info}

    Image.fromarray(pan_format).save(os.path.join(panoptic_folder, out_file_name))

    pbar.update(1)

    return {"image": image, "annotation": annotation}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-folder",
        dest="cityscapesPath",
        help="path to the Cityscapes dataset 'gtFine' folder",
        default="./cityscapes/gtFine",
        type=str,
    )
    parser.add_argument(
        "--output-folder",
        dest="outputFolder",
        help="path to the output folder.",
        default=None,
        type=str,
    )
    parser.add_argument("--use-train-id", action="store_true", dest="useTrainId")
    parser.add_argument(
        "--set-names",
        dest="setNames",
        help="set names to which apply the function to",
        nargs="+",
        default=["val", "train", "test"],
        type=str,
    )
    args = parser.parse_args()

    convert2panoptic(args.cityscapesPath, args.outputFolder, args.useTrainId, args.setNames)


# call the main
if __name__ == "__main__":
    main()
