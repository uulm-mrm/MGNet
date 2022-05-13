#!/usr/bin/env python3
import pickle as pkl
import sys

import torch


def convert_key(k, prefix=""):
    if "layer" not in k:
        k = "stem." + k
    for t in [1, 2, 3, 4]:
        k = k.replace("layer{}".format(t), "res{}".format(t + 1))
    for t in [1, 2, 3]:
        k = k.replace("bn{}".format(t), "conv{}.norm".format(t))
    k = k.replace("downsample.0", "shortcut")
    k = k.replace("downsample.1", "shortcut.norm")
    k = prefix + "." + k
    return k


if __name__ == "__main__":
    newmodel = {}

    input_backbone = sys.argv[1]
    obj = torch.load(input_backbone, map_location="cpu")
    for k in list(obj.keys()):
        old_k = k
        k = convert_key(k, "backbone")
        print(old_k, "->", k)
        newmodel[k] = obj.pop(old_k).detach().numpy()

    input_pose_encoder = sys.argv[2]
    obj = torch.load(input_pose_encoder, map_location="cpu")
    for k in list(obj.keys()):
        old_k = k
        k = convert_key(k, "pose_encoder")
        print(old_k, "->", k)
        if k == "pose_encoder.stem.conv1.weight":
            print("Adapting first conv weight for pose encoder")
            v = torch.cat([obj.pop(old_k)] * 3, 1) / 3
            newmodel[k] = v.detach().numpy()
        else:
            newmodel[k] = obj.pop(old_k).detach().numpy()

    res = {"model": newmodel, "__author__": "torchvision", "matching_heuristics": True}

    with open(sys.argv[3], "wb") as f:
        pkl.dump(res, f)
    if obj:
        print("Unconverted keys:", obj.keys())
