#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

mkdir -p $SCRIPT_DIR/output

echo "Download ImageNet weights..."
mkdir -p $SCRIPT_DIR/weights
wget https://dl.fbaipublicfiles.com/semiweaksupervision/model_files/semi_weakly_supervised_resnet18-118f1556.pth -O $SCRIPT_DIR/weights/backbone.pth
wget https://download.pytorch.org/models/resnet18-f37072fd.pth -O $SCRIPT_DIR/weights/pose_encoder.pth
python3 $SCRIPT_DIR/tools/convert-torchvision-to-mgnet.py $SCRIPT_DIR/weights/backbone.pth $SCRIPT_DIR/weights/pose_encoder.pth $SCRIPT_DIR/weights/imagenet_weights.pkl
echo "Done downloading ImageNet weights"
