# MonocularDepth-Using-LightFields
This repository contains the code used in my dissertation entitled, "Using Light Fields to Enable Deep Monocular Depth Estimation". The original code and models used are taken and modified from various authors. These are all referenced below. The abstract of the dissertation is also listed below. 

The dissertation can be found [here](MCS_Dissertation___Niall_Hunt.pdf).

# Running this evaluation
This repo is **NOT** in a runnable state. Each model has a unique set of requirements to run. It is recommended to follow the instructions outlined by each model individually. The models used in this study are linked below. 

To use the HCI evaluation code (also linked below) the predicted depth maps should be saved as an array rather than an image. This allows us to convert the depth map to a disparity map and then to the required PFM file format. The code I used for this is provided, however, it is still advisable to use the original HCI evaluation code and modify that as necessary for your use case. 

# Abstract
Knowing the depth of a scene is a vital part of many computer vision applications, ranging from autonomous driving to augmented reality. Monocular depth estimation is the technique of predicting the depth of a scene given a single image. This ill-posed problem is increasingly being tackled with end-to-end neural networks thanks to advancements in machine learning. Deep learning techniques require large datasets with accurate ground truth depth maps for training. There are many features within an image that are used to decipher depth, these include: texture, object size and defocused blur. The large datasets that are provided for monocular depth estimation by nature do not include all of these features. 

This project studies the state of the art in monocular depth estimation and their ability to generalise to an unseen dataset that includes images with features across all-in-focus images and images with a defocus blur. This dataset consists of various four-dimensional light fields which unlike traditional two-dimensional images, capture both the spatial intensity and angular direction of light rays. We use these light fields to create three distinct datasets (all-in-focus, front focus and back focus), using a shift-and-sum algorithm. These datasets are used to study and evaluate the performance of the chosen monocular depth estimation techniques. 

In this dissertation we present the findings of the study, showing that despite the advancements made in monocular depth estimation, there is still a large gap in performance in comparison to other depth estimation techniques and that the performance of the state of the art cannot be predicted across the three datasets.

# Models used
## DenseDepth
### Paper
Ibraheem Alhashim and Peter Wonka. High quality monocular depth estimation via transfer learning. _arXiv e-prints_, abs/1812.11941, 2018. [Link](https://arxiv.org/abs/1812.11941)
### Code
https://github.com/ialhashim/DenseDepth

## GDN-Pytorch
### Paper
Minsoo Song and Wonjun Kim. Depth estimation from a single image using guided deep network. _IEEE Access_, 7:142595–142606, 2019. [Link](https://ieeexplore.ieee.org/abstract/document/8854079/)
### Code
https://github.com/tjqansthd/GDN-Pytorch

## Monodepth2
### Paper
Clement Godard, Oisin Mac Aodha, Michael Firman, and Gabriel Brostow. Digging into self-supervised monocular depth estimation. In _2019 IEEE/CVF International Conference on Computer Vision (ICCV)_, pages 3827–3837, 2019. [Link](https://arxiv.org/abs/1806.01260)
### Code
https://github.com/nianticlabs/monodepth2

## Defocus-Net
### Paper
Maxim Maximov, Kevin Galim, and Laura Leal-Taixe. Focus on defocus: Bridging the synthetic to real domain gap for depth estimation. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_, June 2020. [Link](https://openaccess.thecvf.com/content_CVPR_2020/html/Maximov_Focus_on_Defocus_Bridging_the_Synthetic_to_Real_Domain_Gap_CVPR_2020_paper.html)
### Code
https://github.com/dvl-tum/defocus-net

# HCI dataset and evaluation
### Papers
Katrin Honauer, Ole Johannsen, Daniel Kondermann, and Bastian Goldluecke. A dataset and evaluation methodology for depth estimation on 4d light fields. In _Asian Conference on Computer Vision_. Springer, 2016. [Link](http://lightfield-analysis.net/benchmark/paper/lightfield_benchmark_accv_2016.pdf)

Ole Johannsen, Katrin Honauer, Bastian Goldluecke, Anna Alperovich, Federica Battisti, Yunsu Bok, Michele Brizzi, Marco Carli, Gyeongmin Choe, Maximilian Diebold, Marcel Gutsche, Hae-Gon Jeon, In So Kweon, Alessandro Neri, Jaesik Park, Jinsun Park, Hendrik Schilling, Hao Sheng, Lipeng Si, Michael Strecke, Antonin Sulc, Yu-Wing Tai, Qing Wang, Ting-Chun Wang, Sven Wanner, Zhang Xiong, Jingyi Yu, Shuo Zhang, and Hao Zhu. A taxonomy and evaluation of dense light field depth estimation algorithms. In _Conference on Computer Vision and Pattern Recognition - LF4CV Workshop_, 2017. [Link](http://lightfield-analysis.net/benchmark/paper/survey_cvprw_lf4cv_2017.pdf)

### Code
https://github.com/lightfield-analysis/evaluation-toolkit

### Website
https://lightfield-analysis.uni-konstanz.de/
