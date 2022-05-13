FROM nvcr.io/nvidia/pytorch:21.12-py3

ENV DEBIAN_FRONTEND="noninteractive" \
    TERM="xterm"
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install apt packages
RUN apt-get update && apt-get install -y -qq --no-install-recommends  \
	build-essential \
	isort \
	meld \
	zsh \
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

# Install python dependencies
RUN /opt/conda/bin/pip install --upgrade pip
RUN /opt/conda/bin/pip install cityscapesscripts
RUN /opt/conda/bin/pip install git+https://github.com/cocodataset/panopticapi.git
RUN /opt/conda/bin/pip install git+https://github.com/facebookresearch/fvcore
RUN /opt/conda/bin/pip install inplace-abn
RUN /opt/conda/bin/pip install open3d
RUN /opt/conda/bin/pip install opencv-python==4.1.2.30  # Newer versions have a bug with imshow calls inside docker
RUN /opt/conda/bin/pip install pickle5
RUN /opt/conda/bin/pip install Shapely
RUN /opt/conda/bin/pip install setuptools==59.5.0  # Newer versions have a bug with torch tensorboard

# Install onnx-tensorrt
# onnx-tensorrt
RUN cd /tmp \
      && git clone --branch "v3.5.0" https://github.com/google/protobuf.git \
      && cd protobuf \
      && git submodule update --init --recursive \
      && ./autogen.sh \
      && ./configure \
      && make -j8 \
      && make check \
      && make install \
      && ldconfig \
      # cleanup
      && cd /tmp/ \
      && rm -rf protobuf

RUN cd /tmp \
      && git clone --branch "release/8.2-GA" https://github.com/onnx/onnx-tensorrt.git \
      && cd onnx-tensorrt \
      && git submodule update --init --recursive \
      && mkdir build && cd build \
      && cmake .. \
      && make install -j \
      # cleanup
      && cd /tmp/ \
      && rm -rf onnx-tensorrt


# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Detectron2 env vars
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
ENV FVCORE_CACHE="/tmp"

# Install detectron2
RUN cd /opt && \
    git clone https://github.com/facebookresearch/detectron2 && \
    /opt/conda/bin/pip install -e detectron2

# Install MGNet
RUN cd /opt && \
    git clone https://github.com/markusschoen/MGNet.git && \
    /opt/conda/bin/pip install -e MGNet
