# CD-Net
## Introduction
This repository contains the official pytorch implementation of the paper ["Measuring Perceptual Color Differences of Smartphone Photographs"](https://arxiv.org/abs/2205.13489) by Zhihua Wang, Keshuo Xu, Yang Yang, Jianlei Dong, Shuhang Gu, Lihao Xu, Yuming Fang and Kede Ma, IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023. 

Measures for visual color differences (CDs) are pivotal in hardware and software upgrading of modern smartphone photography. We firstly construct currently the largest database for visual CDs for smartphone photography. We then conduct a large-scale psychophysical experiment to gather perceptual CDs of 30,000 image pairs in a carefully controlled laboratory environment. Based on the newly established dataset, we make one of the first attempts to construct an end-to-end learnable CD formula based on a lightweight neural network, as a generalization of several previous metrics. Extensive experiments demonstrate that the optimized formula outperforms 33 existing CD measures by a large margin, offers reasonable local CD maps without the use of dense supervision, generalizes well to homogeneous color patch data.

## Dataset
[Our database](https://ieeexplore.ieee.org/abstract/document/9897498)(SPCD) consists of 15335 natural images:
1) captured by six latest flagship smartphones
2) altered by Photoshop®
3) post-processed by built-in filters of smartphones
4) reproduced with incorrect color profiles
We conduct a large-scale psychophysical experiment to gather perceptual CDs of 30,000 image pairs in a carefully controlled laboratory environment.
## Prerequisites
* python 3.10
* pytorch 1.12.0
* ``pip install -r requirements.txt``
```sh
Python>=3.6
Pytorch>=1.0
cnn_finetune
```
Go check them out if you don't have them locally installed.

======== SPCD Database ========


You can download via [BaiduDisk](https://pan.baidu.com/s/18bzu-qhpMW3PqLTlVdoZRQ?pwd=txeh) or [Google Drive](https://drive.google.com/drive/folders/1Wh9fcDPviZcYWqCpXvnsJux1mnZ5WkCf?usp=share_link).

======== Kaggle Competetion ========

We also host a [Community Prediction Competition](https://www.kaggle.com/competitions/visual-color-difference-evaluation).


## Usage
```sh
$ git clone https://github.com/hellooks/CDnet
```

## Citation
If you find the repository helpful in your resarch, please cite the following papers.
```sh
@article{wang2022measuring,
  title={Measuring Perceptual Color Differences of Smartphone Photography},
  author={Wang, Zhihua and Xu, Keshuo and Yang, Yang and Dong, Jianlei and Gu, Shuhang and Xu, Lihao and Fang, Yuming and Ma, Kede},
  journal={arXiv preprint arXiv:2205.13489},
  year={2022}
}
```
```sh
@inproceedings{xu2022database,
  title={A Database of Visual Color Differences of Modern Smartphone Photography},
  author={Xu, Keshuo and Wang, Zhihua and Yang, Yang and Dong, Jianlei and Xu, Lihao and Fang, Yuming and Ma, Kede},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={3758--3762},
  year={2022},
  organization={IEEE}
}
```
