# CD-Net
This is the repository of paper [Measuring Perceptual Color Differences of Smartphone Photography](https://arxiv.org/abs/2205.13489)


## Background
Measures for visual color differences (CDs) are pivotal in hardware and software upgrading of modern smartphone photography. Towards this goal, we construct currently the largest database for visual CDs for smartphone photography. [Our database](https://ieeexplore.ieee.org/abstract/document/9897498)(SPCD) consists of 15335 natural images:
1) captured by six latest flagship smartphones
2) altered by Photoshop®
3) post-processed by built-in filters of smartphones
4) reproduced with incorrect color profiles

Moreover, we conduct a large-scale psychophysical experiment to gather visual CDs of 30000 image pairs from 20 human subjects in a well-designed laboratory environment. We propose a learning-based and end-to-end optimized CD method based on the proposed dataset, which generalizes CIELAB-based metrics and delivers superior CD assessment performance in the presence of misalignment.



## Highlights

======== Pytorch ========

Requirements:
```sh
Python>=3.6
Pytorch>=1.0
cnn_finetune
```
Go check them out if you don't have them locally installed.

======== Download ========

SPCD Database:

[BaiduDisk](https://pan.baidu.com/s/18bzu-qhpMW3PqLTlVdoZRQ?pwd=txeh)

[Google Drive](https://drive.google.com/drive/folders/1Oox6eQq_N9rrEF0uUeQexd7ANbAgkLW7?usp=share_link)

======== Kaggle Competetion ========

We also host a [Community Prediction Competition](https://www.kaggle.com/competitions/spcd-database).


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
