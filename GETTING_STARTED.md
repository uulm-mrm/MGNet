# Getting Started

This document provides a brief introduction on how to use MGNet.

## Inference Demo with Pre-trained Models

1. Pick a model and its config file, for example, configs/MGNet-Cityscapes-Fine.yaml. 
2. We provide demo.py that is able to demo builtin configs. Run it with:

```shell
python tools/demo.py --config-file /path/to/config_file --input /path/to/image_file --calibration-file /path/to/calibration_file --opts MODEL.WEIGHTS /path/to/checkpoint_file
```

The configs are made for training, therefore we need to specify MODEL.WEIGHTS to a model from model zoo for evaluation. This command will run the inference and show visualizations.

For details of the command line arguments, see `python tools/demo.py -h` or look at its source code
to understand its behavior. Some common arguments are:
* To run on a video, replace `--input files` with `--video-input video.mp4`.
* To show 3d point clouds using open3d, add the `--show-pcl` option. To change the open3d viewpoint, set the `--pcl-view-point /path/to/json_file` option. You find example json files for Cityscapes and KITTI in the dataset directory.
* To save outputs to a directory (for images) or a file (for video), use `--output`.

If you use docker and get the error that matplotlib cannot render images, you might have to enable root access to the XServer by running `xhost +local:root`.

## Training & Evaluation in Command Line

### Training

We provide a script `train_net.py`, that is made to train all the configs provided in MGNet.

To train a model with "train_net.py", first setup the corresponding datasets following [datasets/README.md](./datasets/README.md) and download the pretrained ImageNet weights using `bash ./initialize.sh`, then run:
```shell
python tools/train_net.py --num-gpus 4 --config-file /path/to/config_file
```
Note, that all configs except MGNet-Cityscapes-Fine.yaml need pseudo labels for training (see below). All configs are configured for 4 GPU training. We use 4 NVIDIA RTX 2080Ti GPUs.

### Evaluation

To evaluate a model's performance, use
```shell
python tools/train_net.py --num-gpus 4 --config-file /path/to/config_file --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
For more options, see `python tools/train_net.py -h`.

## Pseudo Label Generation

We provide a script `generate_pseudo_labels.py`, that is made to generate pseudo labels on video sequence datasets using a pretrained model.

To generate pseudo labels, run
```shell
python tools/generate_pseudo_labels.py --num-gpus 4 --config-file /path/to/config_file MODEL.WEIGHTS /path/to/checkpoint_file
```
with either configs/MGNet-Cityscapes-PseudoLabelGeneration.yaml or configs/MGNet-KITTI-Eigen-PseudoLabelGeneration.yaml and a pretrained model on Cityscapes fine dataset.

## Reproduce Results
Use the following steps to reproduce results on Cityscapes and KITTI. 
Note that training runs, especially on Cityscapes, have a large variance, hence it may take multiple runs to reproduce the reported results.
1. Download pretrained ImageNet weights if not already present. Run
```shell
bash ./initialize.sh
```
2. Train a model on Cityscapes-Fine dataset. Run 
```shell
python tools/train_net.py --num-gpus 4 --config-file ./configs/MGNet-Cityscapes-Fine.yaml
```
2. Generate pseudo labels for the Cityscapes-VideoSequence dataset. Run
```shell
python tools/generate_pseudo_labels.py --num-gpus 4 --config-file ./configs/MGNet-Cityscapes-PseudoLabelGeneration.yaml MODEL.WEIGHTS /path/to/cityscapes_fine_checkpoint_file
```
3. Train a model on the Cityscapes-VideoSequence dataset using the Cityscapes-Fine trained model as initialization. Run
```shell
python tools/train_net.py --num-gpus 4 --config-file ./configs/MGNet-Cityscapes-VideoSequence.yaml MODEL.WEIGHTS /path/to/cityscapes_fine_checkpoint_file
```
4. Generate pseudo labels for the KITTI-Eigen dataset. Run
```shell
python tools/generate_pseudo_labels.py --num-gpus 4 --config-file ./configs/MGNet-KITTI-Eigen-PseudoLabelGeneration.yaml MODEL.WEIGHTS /path/to/cityscapes_fine_checkpoint_file
```
6. Train a model on the KITTI-Eigen-Zhou dataset using the Cityscapes-Fine trained model as initialization. Run
```shell
python tools/train_net.py --num-gpus 4 --config-file ./configs/MGNet-KITTI-Eigen-Zhou.yaml MODEL.WEIGHTS /path/to/cityscapes_fine_checkpoint_file
```
## Data Visualization

We provide a script `visualize_data.py`, that is made to visualize data loaded into the model during training time. 
Hence, images and labels will be visualized with training augmentations. This is useful e.g. to inspect the quality of pseudo labels.

To run the script, use
```shell
python tools/visualize_data.py --config-file /path/to/config_file
```
The dataset to visualize is based on the passed config file, e.g. Cityscapes pseudo labels will be visualized by using configs/MGNet-Cityscapes-VideoSequence.yaml

## ONNX/TensorRT Export

We provide a script `onnx_trt_export.py`, that is made to export a trained model to onnx, convert it to TensorRT, and export the postprocessing module to TorchScript. 
This step is optional and only necesseray, if you want to run the optimized model with TensorRT. See [trt_inference/README.md](trt_inference/README.md) for details.

To run the script, use
```shell
python tools/onnx_trt_export.py --config-file /path/to/config_file MODEL.WEIGHTS /path/to/checkpoint_file
```

The exported TensorRT and TorchScript models will be stored in the checkpoint configs OUTPUT_DIR.