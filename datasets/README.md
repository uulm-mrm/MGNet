# Use Builtin Datasets

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

MGNet has builtin support for the Cityscapes and KITTI-Eigen dataset.
The datasets are assumed to exist in a directory specified by the environment variable `DETECTRON2_DATASETS`.
Under this directory, MGNet will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  cityscapes/
  kitti_eigen/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

## Cityscapes

Go to the # [Cityscapes website](https://www.cityscapes-dataset.com/) and download the following dataset parts (You have to create an account to be able to download the zip files):
```shell
leftImg8bit_trainvaltest.zip
gtFine_trainvaltest.zip
disparity_trainvaltest.zip
camera_trainvaltest.zip
leftImg8bit_sequence_trainvaltest.zip
disparity_sequence_trainvaltest.zip
```
Extract all files and generate the panoptic dataset using the preparation script `python prepare_cityscapes.py`.

### Expected dataset structure for Cityscapes:
```
cityscapes/
  camera/
    train/
      aachen/
        camera.json
    val/
    test/
  disparity/
    train/
      aachen/
        disparity.png
    val/
    test/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
    # below are generated Cityscapes panoptic annotation
    cityscapes_panoptic_train.json
    cityscapes_panoptic_train/
    cityscapes_panoptic_val.json
    cityscapes_panoptic_val/
    cityscapes_panoptic_test.json
    cityscapes_panoptic_test/
  leftImg8bit/
    train/
    val/
    test/
  leftImg8bit_sequence/
    train/
```

## KITTI-Eigen

Please follow the instructions in [packnet-sfm](https://github.com/TRI-ML/packnet-sfm#kitti) to download the KITTI-Eigen dataset in the correct structure.
Note that since KITTI-Eigen does not provide panoptic annotations, MGNet training on KITTI-Eigen requires pseudo_labels using `tools/generate_pseudo_labels.py` with a pretrained Cityscapes model. 
See [getting started](../GETTING_STARTED.md) for instructions on how to train a model on KITTI-Eigen.

### Expected dataset structure for KITTI:
```
kitti_eigen/
  2011_09_26/
    2011_09_26_drive_0001_sync/
      image_02/
        data/
          0000000000.png
          ...
        cam.txt
        poses.txt
        timestamps.txt
      image_03/
      oxts/
      proj_depth/
        groundtruth/
          image_02/
            0000000005.png
            ...
          image_03/
        velodyne/
    2011_09_26_drive_0002_sync/
    ...
    calib_cam_to_cam.txt
    calib_imu_to_velo.txt
    calib_velo_to_cam.txt
  2011_09_28/
  2011_09_29/
  2011_09_30/
  2011_10_03/
  data_splits/
    eigen_test_files.txt
    eigen_zhou_files.txt
    ...
```