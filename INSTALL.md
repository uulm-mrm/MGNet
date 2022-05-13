# Installation
## Docker (Recommended)
The recommended way for using MGNet is from within a Docker container.
Only the NVIDIA driver (version ≥ 510), Docker (https://docs.docker.com/get-docker/), and nvidia-docker (https://github.com/NVIDIA/nvidia-docker) have to be installed on your host system for using these images.

We provide two Dockerfiles. A base Dockerfile, which contains all the necessary dependencies to use MGNet, and a develop.Dockerfile, which can be used for adding new features to MGNet. 

To build the base `mgnet:latest` image, run 
```bash
cd docker
./build_docker.sh
```

Optionally, after building `mgnet:latest`, you can build the development container `mgnet-dev:latest` by running `./build_docker.sh -d`.

To run a container, call `./docker/run_docker.sh`, which starts a new container in interactive bash mode. The project source code is located in `/opt/MGNet`.

To run a development container, call `./docker/run_docker.sh -d`.

See `./docker/run_docker.sh -h` for all docker run options.

## Local
### Requirements

* Linux with Python ≥ 3.6
* NVIDIA driver 510
* NVIDIA cuda 11.5
* NVIDIA cuDNN 8.3.1
* Optional for TensorRT inference: NVIDIA TensorRT 8.2.1.8

### Install MGNet

Call the following commands from the root directory of this repository to install MGNet along with all necessary dependencies.
We recommend using a `virtualenv` for this.

```bash
# System requirements
sudo apt update
sudo apt install -y  \
	build-essential \
	isort \
	libsm6 \
	libxext6 \
	libxrender-dev \
	libcap-dev \
	libgl1-mesa-glx \
	libusb-1.0-0 \
	libglvnd-dev \
	libgl1-mesa-dev \
	libegl1-mesa-dev \
	libx11-6 \
	libgtk2.0-dev

# (Optional) Create and source virtual environment
python3 -m venv venv
source venv/bin/activate

# Python dependencies
python3 -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
python3 -m pip install -r requirements.txt

# Other dependencies
mkdir deps
cd deps
# Install detectron2
git clone https://github.com/facebookresearch/detectron2
python3 -m pip install -e detectron2

# (Optional) Install TensorRT libs
git clone --branch 8.2.1 https://github.com/nvidia/TensorRT \
    && cd TensorRT \
    && git submodule update --init --recursive \
    && mkdir -p build && cd build \
    && cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DTRT_LIB_DIR=/lib/x86_64-linux-gnu -DTRT_OUT_DIR=`pwd`/out \
    && make -j$(nproc) \
    && sudo make install \
    && cd ../.. \
    && rm -rf TensorRT

# (Optional) Install protobuf and onnx-tensorrt
git clone --branch "v3.5.0" https://github.com/google/protobuf.git \
    && cd protobuf \
    && git submodule update --init --recursive \
    && ./autogen.sh \
    && ./configure \
    && make -j8 \
    && make check \
    && sudo make install \
    && sudo ldconfig \
    && cd .. \
    && rm -rf protobuf
git clone --branch "release/8.2-GA" https://github.com/onnx/onnx-tensorrt.git \
    && cd onnx-tensorrt \
    && git submodule update --init --recursive \
    && mkdir build && cd build \
    && cmake .. \
    && sudo make install -j \
    && cd ../.. \
    && rm -rf onnx-tensorrt

# (Optional) Install libtorch from source
git clone --branch v1.11.0 --recurse-submodule https://github.com/pytorch/pytorch.git \
    && mkdir pytorch-build \
    && cd pytorch-build \
    && export CUDACXX=/usr/local/cuda/bin/nvcc \
    && cmake -DBUILD_SHARED_LIBS:BOOL=ON -DCMAKE_BUILD_TYPE:STRING=Release -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../pytorch-install ../pytorch \
    && cmake --build . --target install \
    && cd .. \
    && rm -rf pytorch pytorch-build

cd ..

# Install MGNet in editable mode
python3 -m pip install -e .
```
