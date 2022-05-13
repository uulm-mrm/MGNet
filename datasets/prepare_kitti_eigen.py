#!/usr/bin/python3
# This script is used in tools/generate_pseudo_labels.py to convert the generated pseudo labels
# for the KITTI-Eigen dataset to COCO-style panoptic segmentation format (http://cocodataset.org/#format-data).  # noqa
import json
import os
import sys
from functools import partial
from multiprocessing.dummy import Pool
from pathlib import Path

import numpy as np
from datasets.labels_cityscapes import id2label, labels

# Image processing
from PIL import Image

# The main method
from tqdm import tqdm

__all__ = ["convert2panoptic"]


def convert2panoptic(kitti_path, image_split_file):
    categories = []
    for label in labels:
        if label.ignoreInEval:
            continue
        categories.append(
            {
                "id": int(label.id),
                "name": label.name,
                "color": label.color,
                "supercategory": label.category,
                "isthing": 1 if label.hasInstances else 0,
            }
        )

    with open(image_split_file) as f:
        files = f.read().splitlines()
    files = [x.split(" ")[0].replace("image_", "label_") for x in files]
    if not files:
        print("Error: Did not find any files in file list {}.".format(image_split_file))
        sys.exit(-1)

    print("Converting {} annotation files for file list {}.".format(len(files), image_split_file))

    output_base_file = "{}_panoptic".format(Path(image_split_file).stem)
    out_file = os.path.join(
        kitti_path, "panoptic_pseudo_labels", "{}.json".format(output_base_file)
    )
    print("Json file with the annotations in panoptic format will be saved in {}".format(out_file))
    panoptic_folder = os.path.join(kitti_path, "panoptic_pseudo_labels", output_base_file)
    if not os.path.isdir(panoptic_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(panoptic_folder))
        os.mkdir(panoptic_folder)
    print("Corresponding segmentations in .png format will be saved in {}".format(panoptic_folder))

    pool = Pool()
    global pbar
    pbar = tqdm(
        total=len(files), desc=f"Convert {image_split_file} to COCO format", file=sys.stdout
    )
    convert_single_file = partial(
        _convert_single_file, kitti_path=kitti_path, panoptic_folder=panoptic_folder
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


def _convert_single_file(file, kitti_path, panoptic_folder):
    image_file = os.path.join(kitti_path, file)
    original_format = np.array(Image.open(image_file))

    image_id = file[:-4]
    out_file_name = file
    # image entry, id for image is its filename without extension
    image = {
        "id": image_id,
        "width": int(original_format.shape[1]),
        "height": int(original_format.shape[0]),
        "file_name": file.replace("label_", "image_"),
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
        category_id = label_info.id
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

    os.makedirs(os.path.join(panoptic_folder, os.path.dirname(out_file_name)), exist_ok=True)
    Image.fromarray(pan_format).save(os.path.join(panoptic_folder, out_file_name))

    pbar.update(1)

    return {"image": image, "annotation": annotation}
