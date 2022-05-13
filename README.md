# MGNet: Monocular Geometric Scene Understanding for Autonomous Driving 

<a target="_blank">
<img src="/media/result_kitti.gif"/>
</a>

This repository contains the official implementation of our ICCV 2021 paper [MGNet: Monocular Geometric Scene Understanding for Autonomous Driving](https://openaccess.thecvf.com/content/ICCV2021/papers/Schon_MGNet_Monocular_Geometric_Scene_Understanding_for_Autonomous_Driving_ICCV_2021_paper.pdf). 

This is a re-implementation based on [detectron2](https://github.com/facebookresearch/detectron2), hence results differ slightly compared to the ones reported in the paper.

## Installation

See [INSTALL.md](INSTALL.md) for instructions on how to prepare your environment to use MGNet.

## Usage

See [datasets/README.md](datasets/README.md) for instructions on how to prepare datasets for MGNet.

See [GETTING_STARTED.md](GETTING_STARTED.md) for instructions on how to train and evaluate models, or run inference on demo images.

See [trt_inference/README.md](trt_inference/README.md) for instructions on how to export trained models to TensorRT and run optimized inference.

## Model Zoo

All models were trained using 4 NVIDIA 2080Ti GPUs.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Name</th>
<th valign="bottom">PQ</th>
<th valign="bottom">PQ_St</th>
<th valign="bottom">PQ_Th</th>
<th valign="bottom">Abs Rel</th>
<th valign="bottom">RMSE</th>
<th valign="bottom">δ < 1.25</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
 <tr><td align="left"><a href="configs/MGNet-Cityscapes-Fine.yaml">MGNet Cityscapes Fine</a></td>
<td align="center">54.879</td>
<td align="center">62.524</td>
<td align="center">44.367</td>
<td align="center">0.188</td>
<td align="center">8.439</td>
<td align="center">0.744</td>
<td align="center"><a href="https://drive.google.com/file/d/16iPZJZIvPFxgapJZUaWPKOX3cI0GuX9K/view?usp=sharing">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/MGNet-Cityscapes-VideoSequence.yaml">MGNet Cityscapes Video Sequence</a></td>
<td align="center">55.644</td>
<td align="center">63.140</td>
<td align="center">45.337</td>
<td align="center">0.166</td>
<td align="center">7.984</td>
<td align="center">0.794</td>
<td align="center"><a href="https://drive.google.com/file/d/1jegknv6zYf5teaE2Utq-UQjBIlSiG3DX/view?usp=sharing">model</a></td>
</tr>
 <tr><td align="left"><a href="configs/MGNet-KITTI-Eigen-Zhou.yaml">MGNet KITTI Eigen Zhou</a></td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center">0.095</td>
<td align="center">3.788</td>
<td align="center">0.897</td>
<td align="center"><a href="https://drive.google.com/file/d/1il4oe3oPfkpScVIq5nspnAmcnKfCYbhX/view?usp=sharing">model</a></td>
</tr>
</tbody></table>

## Reference

Please use the following citations when referencing our work:

**MGNet: Monocular Geometric Scene Understanding for Autonomous Driving (ICCV 2021)** \
*Markus Schön, Michael Buchholz and Klaus Dietmayer*, [**[paper]**](https://openaccess.thecvf.com/content/ICCV2021/papers/Schon_MGNet_Monocular_Geometric_Scene_Understanding_for_Autonomous_Driving_ICCV_2021_paper.pdf), [**[video]**](https://www.youtube.com/watch?v=GXdQNtVQYmY)

```
@InProceedings{Schoen_2021_ICCV,
    author    = {Sch{\"o}n, Markus and Buchholz, Michael and Dietmayer, Klaus},
    title     = {MGNet: Monocular Geometric Scene Understanding for Autonomous Driving},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {15804-15815}
}
```
## Acknowledgement

We used and modified code parts from other open source projects, we especially like to thank the authors of:

- [detectron2](https://github.com/facebookresearch/detectron2)
- [packnet-sfm](https://github.com/TRI-ML/packnet-sfm)
- [TorchSeg](https://github.com/ycszen/TorchSeg)