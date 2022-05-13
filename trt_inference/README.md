# TensorRT inference

This directory contains C++ code to run an exported MGNet model with TensorRT optimized performance.
Please follow [ONNX/TensorRT Export](../GETTING_STARTED.md#onnxtensorrt-export) on how to export pretrained models.

Note, that the code is tailored for Cityscapes models. To deploy models trained on other datasets, adapt the Code in [main.cpp](./main.cpp). The code comments will guide you on what needs to be updated.

## Build

To build the TensorRT code, make sure TensorRT and libtorch is installed and run
```shell
mkdir build
cd build
cmake ..
make -j
```
If you followed the [Local Installation](../INSTALL.md#local) guide, you have to update [CMakeLists.txt](./CMakeLists.txt) to use the correct path for `find_package(Torch)`.

# Run

To run inference, call
```shell
./trt_inference /path/to/model_plan_file /path/to/postprocessing_pt_file /path/to/image_file /path/to/calibration_file
```
Note, that the calibration file should be in Cityscapes format and contain at least the intrinsic parameters and the extrinsic z component.
