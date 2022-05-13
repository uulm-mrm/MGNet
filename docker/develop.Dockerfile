ARG TAG=latest
FROM mgnet:${TAG}

# Install clang-format for linter runs
RUN apt-get update && apt-get install -y -qq --no-install-recommends clang-format python3-setuptools

# create a non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --system --gid ${GROUP_ID} appuser
RUN useradd --create-home --no-log-init --system --uid ${USER_ID} --gid ${GROUP_ID} --groups sudo appuser
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER appuser
WORKDIR /home/appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# Install fvcore
RUN git clone https://github.com/facebookresearch/fvcore && \
	pip install --user -e fvcore

# Install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 && \
	pip install --user -e detectron2

# Install MGNet
RUN git clone https://github.com/markusschoen/MGNet.git && \
	pip install --user -e MGNet

WORKDIR /home/appuser/MGNet
