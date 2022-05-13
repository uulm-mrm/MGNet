from typing import Dict, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling import BACKBONE_REGISTRY
from inplace_abn import InPlaceABNSync

__all__ = [
    "MGNetDecoder",
    "MGNetHead",
    "PoseCNN",
    "FastGlobalAvgPool2d",
    "GlobalContextModule",
    "AttentionRefinementModule",
    "FeatureFusionModule",
]


class MGNetDecoder(nn.Module):
    """
    MGNetDecoder module combines different feature map scales from an encoder
    using AttentionRefinementModules and FeatureFusionModule
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        common_stride: int,
        arm_channels: List[int],
        refine_channels: List[int],
        ffm_channels: int,
        init_method: str = "default",
    ):
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride, reverse=True)

        # fmt: off
        self.in_features = [k for k, v in input_shape]
        in_channels = [x[1].channels for x in input_shape]
        self.common_stride = common_stride
        # fmt: on

        assert len(arm_channels) == 2, "arm_channels have to be a list of ints with length 2!"
        assert len(refine_channels) == 2, "refine_channels have to be a list of ints with length 2!"

        self.arms = torch.nn.ModuleList(
            [
                AttentionRefinementModule(in_channels[0], arm_channels[0], init_method=init_method),
                AttentionRefinementModule(in_channels[1], arm_channels[1], init_method=init_method),
            ]
        )
        self.refines = torch.nn.ModuleList(
            [
                Conv2d(
                    arm_channels[0],
                    refine_channels[0],
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    norm=InPlaceABNSync(refine_channels[0], momentum=0.01, group=dist.group.WORLD),
                ),
                Conv2d(
                    arm_channels[1],
                    refine_channels[1],
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    norm=InPlaceABNSync(refine_channels[1], momentum=0.01, group=dist.group.WORLD),
                ),
            ]
        )
        self.ffm = FeatureFusionModule(
            in_channels[2] + refine_channels[1], ffm_channels, init_method=init_method
        )
        if init_method == "xavier":
            mgnet_xavier_fill(self.refines[0])
            mgnet_xavier_fill(self.refines[1])

    def forward(self, features):
        feature_list = [features[x] for x in self.in_features]
        msc_features = []
        last_fm = features["global_context"]
        for i, (fm, arm, refine) in enumerate(zip(feature_list[:2], self.arms, self.refines)):
            fm = arm(fm)
            fm += last_fm
            msc_features.append(fm)
            last_fm = F.interpolate(fm, size=(feature_list[i + 1].size()[2:]), mode="nearest")
            last_fm = refine(last_fm)

        y = self.ffm(feature_list[2], last_fm)
        return y, msc_features


class MGNetHead(nn.Module):
    """
    MGNetHead module using two sequential Conv2d layers.
    Creates an output feature map with num_classes channels.
    """

    def __init__(
        self,
        in_channels: int,
        head_channels: int,
        num_classes: int,
        init_method: str = "default",
    ):
        super().__init__()
        self.head = Conv2d(
            in_channels,
            head_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            norm=InPlaceABNSync(head_channels, momentum=0.01, group=dist.group.WORLD),
        )
        self.predictor = nn.Conv2d(head_channels, num_classes, kernel_size=(1, 1), bias=False)
        if init_method == "xavier":
            mgnet_xavier_fill(self.head)
            mgnet_xavier_fill(self.predictor)

    def forward(self, x):
        y = self.head(x)
        y = self.predictor(y)
        return y


class PoseCNN(nn.Module):
    """
    PoseCNN based on ResNet encoder and convolutional decoder. Predicts 6 DOF relative camera poses
    between current frame and num_context_images frames in the video sequence.
    """

    def __init__(self, cfg, num_context_images=2):
        super().__init__()
        self.num_context_images = num_context_images

        # Only the torchvision initialization is working for the pose encoder.
        backbone_name = cfg.MODEL.BACKBONE.NAME
        self.pose_encoder = BACKBONE_REGISTRY.get(backbone_name)(
            cfg, ShapeSpec(channels=(self.num_context_images + 1) * 3)
        )

        self.conv1 = nn.Conv2d(512, 256, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(256, 6 * num_context_images, kernel_size=(1, 1))
        mgnet_xavier_fill(self.conv1)
        mgnet_xavier_fill(self.conv2)
        mgnet_xavier_fill(self.conv3)
        mgnet_xavier_fill(self.conv4)

    def forward(self, image_list):
        # Encode images
        input_features = self.pose_encoder(image_list)

        # Decode features
        out = F.relu_(self.conv1(input_features["res5"]))
        out = F.relu_(self.conv2(out))
        out = F.relu_(self.conv3(out))
        out = self.conv4(out)

        out = out.mean(3).mean(2)
        out = 0.01 * out.view(out.size(0), self.num_context_images, 6)
        return out


class FastGlobalAvgPool2d(nn.Module):
    """
    Fast global avg pooling
    Adopted from https://github.com/mrT23/TResNet/blob/master/src/models/tresnet/layers/avg_pool.py
    """

    def __init__(self, flatten=False):
        super().__init__()
        self.flatten = flatten

    def forward(self, x):
        if self.flatten:
            in_size = x.size()
            return x.view((in_size[0], in_size[1], -1)).mean(dim=2)
        return x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)


class GlobalContextModule(nn.Module):
    """
    Adopted from TorchSeg
    https://github.com/ycszen/TorchSeg/blob/master/model/bisenet/cityscapes.bisenet.R18.speed/network.py  # noqa
    using detectron2 layers, InPlaceABNSync and FastGlobalAvgPool2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_method: str = "default",
    ):
        super().__init__()
        self.global_context = nn.Sequential(
            FastGlobalAvgPool2d(),
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False,
                norm=InPlaceABNSync(out_channels, momentum=0.01, group=dist.group.WORLD),
            ),
        )
        if init_method == "xavier":
            mgnet_xavier_fill(self.global_context[1])

    def forward(self, x):
        y = self.global_context(x)
        y = F.interpolate(y, x.size()[2:], mode="nearest")
        return y


class AttentionRefinementModule(nn.Module):
    """
    Adopted from TorchSeg
    https://github.com/ycszen/TorchSeg/blob/master/furnace/seg_opr/seg_oprs.py
    using detectron2 layers, InPlaceABNSync and FastGlobalAvgPool2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_method: str = "default",
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=InPlaceABNSync(out_channels, momentum=0.01, group=dist.group.WORLD),
        )
        self.channel_attention = nn.Sequential(
            FastGlobalAvgPool2d(),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                norm=InPlaceABNSync(
                    out_channels, momentum=0.01, activation="identity", group=dist.group.WORLD
                ),
            ),
            nn.Sigmoid(),
        )
        if init_method == "xavier":
            mgnet_xavier_fill(self.conv)
            mgnet_xavier_fill(self.channel_attention[1])

    def forward(self, x):
        fm = self.conv(x)
        atten = self.channel_attention(fm)
        fm = fm * atten
        return fm


class FeatureFusionModule(nn.Module):
    """
    Adopted from TorchSeg
    https://github.com/ycszen/TorchSeg/blob/master/furnace/seg_opr/seg_oprs.py
    using detectron2 layers, InPlaceABNSync and FastGlobalAvgPool2d
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        init_method: str = "default",
    ):
        super().__init__()
        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            norm=InPlaceABNSync(out_channels, momentum=0.01, group=dist.group.WORLD),
        )
        self.channel_attention = nn.Sequential(
            FastGlobalAvgPool2d(),
            Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                activation=nn.ReLU(inplace=True),
            ),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(1, 1),
                bias=False,
            ),
            nn.Sigmoid(),
        )
        if init_method == "xavier":
            mgnet_xavier_fill(self.conv)
            mgnet_xavier_fill(self.channel_attention[1])
            mgnet_xavier_fill(self.channel_attention[2])

    def forward(self, fsp, fcp):
        fm = torch.cat([fsp, fcp], dim=1)
        fm = self.conv(fm)
        atten = self.channel_attention(fm)
        fm = fm + fm * atten
        return fm


def mgnet_xavier_fill(module: nn.Module) -> None:
    torch.nn.init.kaiming_normal_(module.weight, a=1)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
