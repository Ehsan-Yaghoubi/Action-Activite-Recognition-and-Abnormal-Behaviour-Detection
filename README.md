### Skeleton based action/activity recognition and abnormal behaviour detection methods

<details>
  
<summary>Open source softwares for pose estimation: (click to expand!)</summary>
  
  [MMPose](https://github.com/open-mmlab/mmpose): an open-source toolbox for pose estimation based on PyTorch.
  
  (2019-2021) [OpenPifPaf](https://github.com/wangzheallen/awesome-human-pose-estimation), [paper CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.html), [arxiv paper](https://arxiv.org/abs/2103.02440): Continuously tested on Linux, MacOS and Windows. [Tutorial](https://openpifpaf.github.io/intro.html)
  
  (2018) [Simple Baseline](https://github.com/microsoft/human-pose-estimation.pytorch), [arxive paper](https://arxiv.org/abs/1804.06208): Pytorch 0.4, python 3.6, Ubuntu 16.04, NVIDIA GPUs.
  
  (2021) [SWAHR-HumanPose](https://github.com/greatlog/SWAHR-HumanPose), [paper CVPR2021](https://arxiv.org/abs/2012.15175): 
  
  (2018) [Real-time 2D Multi-Person Pose Estimation *on CPU*: Lightweight OpenPose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch), [arxiv paper](https://arxiv.org/abs/1811.12004): Ubuntu 16.04, Python 3.6, PyTorch 0.4.1, No GPU needed.
  
  (2019) [VideoPose3D](https://github.com/facebookresearch/VideoPose3D), [paper CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Pavllo_3D_Human_Pose_Estimation_in_Video_With_Temporal_Convolutions_and_CVPR_2019_paper.pdf): Python 3+ , PyTorch >= 0.4.0.
  
  (2020) [VoxelPose:Towards Multi-Camera 3D Human Pose Estimation in Wild Environment](https://github.com/microsoft/voxelpose-pytorch), [paper ECCV2020](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/738_ECCV_2020_paper.php):
  
  (2020) [VIBE: Video Inference for Human Body Pose and Shape Estimation](https://github.com/mkocabas/VIBE), [paper CVPR2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kocabas_VIBE_Video_Inference_for_Human_Body_Pose_and_Shape_Estimation_CVPR_2020_paper.pdf):
  
  (2021) [PoseAug: A Differentiable Pose Augmentation Framework for 3D Human Pose Estimation](https://github.com/jfzhang95/PoseAug), [paper CVPR2021](https://arxiv.org/abs/2105.02465):
  
  (since 2016) [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose), [paper IEEE2019](https://ieeexplore.ieee.org/abstract/document/8765346/), [paper CVPR2017](http://openaccess.thecvf.com/content_cvpr_2017/html/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.html), [paper CVPR2017](http://openaccess.thecvf.com/content_cvpr_2017/html/Simon_Hand_Keypoint_Detection_CVPR_2017_paper.html), [paper CVPR2016](http://openaccess.thecvf.com/content_cvpr_2016/html/Wei_Convolutional_Pose_Machines_CVPR_2016_paper.html): Real-time multi-person system to jointly detect human body, hand, facial, and foot keypoints (in total 135 keypoints) on single images.
  
  (2019) [Cross View Fusion for 3D Human Pose Estimation](https://github.com/microsoft/multiview-human-pose-estimation-pytorch), [paper ICCV2019](https://chunyuwang.netlify.com/img/ICCV_Cross_view_camera_ready.pdf): Uses MPII,  Human36M, and Limb length prior for 3D Pose Estimation. For 2D task they use multiview training on mixed datasets (MPII+H36M) and test on H36M. For 3D task they use multiview test on H36M (based on CPU or GPU).
  
  (2021) [3D Human Pose Estimation with Spatial and Temporal Transformers](https://github.com/zczcwh/PoseFormer), [paper ICCV2021](https://arxiv.org/abs/2103.10455):
  
  (2020) [UniPose: Unified Human Pose Estimation in Single Images and Videos](https://github.com/bmartacho/UniPose), [paper CVPR2020](https://arxiv.org/abs/2001.08095): datasets used are LSP, MPII, PennAction, and BBC Pose.  
  
  (2018) [Integral Human Pose Regression](https://github.com/JimmySuen/integral-human-pose), [paper ICCV2018](http://openaccess.thecvf.com/content_ECCV_2018/html/Xiao_Sun_Integral_Human_Pose_ECCV_2018_paper.html), [paper 2 arxiv](https://arxiv.org/abs/1809.06079): Python 3.6, tested on CentOs7 (Other Linux system is also OK), CUDA  9.0 (least 8.0), PyTorch 0.4.0.
  
  (2021) [Bottom-Up Human Pose Estimation Via Disentangled Keypoint Regression](https://github.com/HRNet/DEKR), [CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Geng_Bottom-Up_Human_Pose_Estimation_via_Disentangled_Keypoint_Regression_CVPR_2021_paper.pdf): python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed.
  
  (2021) [HPRNet: Hierarchical Point Regression for Whole-Body Human Pose Estimation](https://github.com/nerminsamet/HPRNet), [arxiv paper](https://arxiv.org/abs/2106.04269): HPRNet is a bottom-up, one-stage and hierarchical keypoint regression method. PyTorch 1.4.0

  (2020) [DarkPose:Distribution Aware Coordinate Representation for Human Pose Estimation](https://github.com/ilovepose/DarkPose) [paper CVPR2020](http://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.html): 2nd place entry of COCO Keypoints Challenge ICCV 2019. uses the MPII dataset.
  
</details>
  
  
  
  
  
  
  
<details>
<summary>Paper list:</summary>
  
  1.(2016) (RNN) [Deep LSTM + Part-Aware LSTM + NTU RGB+D dataset](https://openaccess.thecvf.com/content_cvpr_2016/html/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.html)

  3.(2016) (RNN) [Spatio-Temporal LSTM with Trust Gates for 3D Human Action Recognition](https://link.springer.com/chapter/10.1007/978-3-319-46487-9_50)

  4.(2017) (RNN) [View Adaptive RNN for High Performance Human Action Recognition From Skeleton Data](https://openaccess.thecvf.com/content_iccv_2017/html/Zhang_View_Adaptive_Recurrent_ICCV_2017_paper.html)

  5.(2017) (CNN) [Two-Stream 3D Convolutional Neural Network for Skeleton-Based Action Recognition](https://arxiv.org/abs/1705.08106)

  6.(2017) (CNN) [Interpretable 3D Human Action Analysis With Temporal Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w20/html/Kim_Interpretable_3D_Human_CVPR_2017_paper.html)

  7.(2017) (CNN) [Enhanced skeleton visualization for view invariant human action recognition](https://www.sciencedirect.com/science/article/pii/S0031320317300936)

  8.(2018) (GCN) [Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition](https://arxiv.org/abs/1801.07455)

  9.(2018) (GCN) [Deep Progressive Reinforcement Learning for Skeleton-Based Action Recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Tang_Deep_Progressive_Reinforcement_CVPR_2018_paper.html)

  10.(2019) (GCN) [Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html)

  11.(2019) (GCN) [Actional-Structural Graph Convolutional Networks for Skeleton-Based Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Actional-Structural_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.html)

  12.(2019) (GCN) [An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Si_An_Attention_Enhanced_Graph_Convolutional_LSTM_Network_for_Skeleton-Based_Action_CVPR_2019_paper.html)

  13.(2019) (GCN) [Skeleton-Based Action Recognition With Directed Graph Neural Networks](https://openaccess.thecvf.com/content_CVPR_2019/html/Shi_Skeleton-Based_Action_Recognition_With_Directed_Graph_Neural_Networks_CVPR_2019_paper.html)

  14.(2020) (GCN) [Skeleton-Based Action Recognition With Shift Graph Convolutional Network](https://openaccess.thecvf.com/content_CVPR_2020/html/Cheng_Skeleton-Based_Action_Recognition_With_Shift_Graph_Convolutional_Network_CVPR_2020_paper.html)

  15.(2020) (GCN) [Context Aware Graph Convolution for Skeleton-Based Action Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Context_Aware_Graph_Convolution_for_Skeleton-Based_Action_Recognition_CVPR_2020_paper.html)

  16.(2020) (GCN) [Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhang_Semantics-Guided_Neural_Networks_for_Efficient_Skeleton-Based_Human_Action_Recognition_CVPR_2020_paper.html)

  (2021) () [Revisiting Skeleton-based Action Recognition](https://arxiv.org/abs/2104.13586) [github](https://github.com/open-mmlab/mmaction2/blob/master/configs/skeleton/posec3d/README.md)

  (2021) () [Memory Attention Networks for Skeleton-Based Action Recognition](https://ieeexplore.ieee.org/abstract/document/9378801)

  (2021) (Transformer) [Spatial Temporal Transformer Network for Skeleton-Based Action Recognition](https://link.springer.com/chapter/10.1007/978-3-030-68796-0_50)

  (2021) () [Quo Vadis, Skeleton Action Recognition?](https://link.springer.com/article/10.1007/s11263-021-01470-y)

  (2021) (GCN) [Symbiotic Graph Neural Networks for 3D Skeleton-based Human Action Recognition and Motion Prediction](https://ieeexplore.ieee.org/abstract/document/9334430/)

  (2021) () [Tripool: Graph triplet pooling for 3D skeleton-based action recognition](https://www.sciencedirect.com/science/article/pii/S0031320321001084)

  (2021) (GCN) [Spatial Temporal Graph Deconvolutional Network for Skeleton-Based Human Action Recognition](https://ieeexplore.ieee.org/abstract/document/9314910)

  (2021) (GCN) [Skeleton-based action recognition using sparse spatio-temporal GCN with edge effective resistance](https://www.sciencedirect.com/science/article/pii/S0925231220317094)

  (2021) (GCN) [Predictively encoded graph convolutional network for noise-robust skeleton-based action recognition](https://link.springer.com/article/10.1007/s10489-021-02487-z)

  (2021) () [Structural Knowledge Distillation for Efficient Skeleton-Based Action Recognition](https://ieeexplore.ieee.org/abstract/document/9351789/)

  (2021) () [Symmetrical Enhanced Fusion Network for Skeleton-based Action Recognition](https://ieeexplore.ieee.org/abstract/document/9319717)

  (2021) (GCN) [Temporal Attention-Augmented Graph Convolutional Network for Efficient Skeleton-Based Human Action Recognition](https://ieeexplore.ieee.org/abstract/document/9412091)

  (2021) () [Rethinking the ST-GCNs for 3D skeleton-based human action recognition](https://www.sciencedirect.com/science/article/pii/S0925231221007153)
  
</details>

<details>
<summary>Datasets:</summary>
  
### Skeleton-based Datasets:

  (2013) [Human3.6M](https://ieeexplore.ieee.org/abstract/document/6682899), 2-dimensional space, 10 action classes, smaller than the next ones, human labelled.

  (2017) [Kinetics-Skeleton](https://arxiv.org/abs/1705.06950 ), 2-dimensional space, 400 action classes, large-scale, pose estimated, (original RGB videos).

  (2016/2019) [NTU RGB+D](https://openaccess.thecvf.com/content_cvpr_2016/html/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.html), [NTU RGB+D 120](https://ieeexplore.ieee.org/abstract/document/8713892/). 56,880 (/57,600) videos with 60 (/120) action classes that provides 3D skeleton and RGB-D data. bounding box annotations

### RGB video-based Datasets:

  (2014) [UCF Sports Action Data Set](https://www.crcv.ucf.edu/data/UCF_Sports_Action.php) - 150 clips with mean length of 6sec, 10 actions, resolution 720x480 

  (2019) [HOLLYWOOD2](https://www.di.ens.fr/~laptev/actions/hollywood2/) - 3669 video clips, 12 action classes and 10 classes of scenes, (20hours from 69 movies) 

  (2012) [UCF-101](https://arxiv.org/abs/1212.0402). 13,320 videos from youtube; 101 action classes, frame-level annotation.

  (2011) [HMDB-51](https://ieeexplore.ieee.org/document/6126543). 7000 videos from youtube, 51 action classes with at least 101 videos, frame-level annotation.

  (2017/2018/2020) [Kinetics-400](https://arxiv.org/abs/1705.06950), [Kinetics-600](https://arxiv.org/abs/1808.01340), [Kinetics-700](https://arxiv.org/abs/2010.10864). With 400 (/600/700) action classes with 400 (/600/700) 10-second videos for each class. Frame-level annotation.

  (2014) [Sports-1M](https://www.cv-foundation.org/openaccess/content_cvpr_2014/html/Karpathy_Large-scale_Video_Classification_2014_CVPR_paper.html).  1,133,158 videos, automaticaly annotated for 487 actions at video-level. Each video has a url to be downloaded from.
  
  (2014) [THUMOS-14](https://www.sciencedirect.com/science/article/pii/S1077314216301710). 18000 videos, 101 action classes. With trimmed training videos and untrimmed test data, and with rame-level annotation.
  
  (2012) [UCF-101-24](https://arxiv.org/abs/1212.0402). Originated from the UCF-101 dataset with 24 action classes annotated at the pixel-level. 
  
  (2013) [J-HMDB-21](https://www.cv-foundation.org/openaccess/content_iccv_2013/html/Jhuang_Towards_Understanding_Action_2013_ICCV_paper.html). Originated from the HMDB-51 dataset with 21 action classes annotated at the pixel-level.
  
  (2018) [AVA](https://research.google.com/ava/), [well-written paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Gu_AVA_A_Video_CVPR_2018_paper.pdf). 430 video clips from movies, 15 minutes each with 900 keyframes, in each of which, the persons were labeled with multiple actions. With bounding box annotation.
  
  (2016/2019) [NTU RGB+D](https://openaccess.thecvf.com/content_cvpr_2016/html/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.html), [NTU RGB+D 120](https://ieeexplore.ieee.org/abstract/document/8713892/). 56,880 (/57,600) videos with 60 (/120) action classes that provides 3D skeleton and RGB-D data. bounding box annotations
</details>



<details>
<summary>Useful Links:</summary>
  
  [Pose estimation](https://github.com/wangzheallen/awesome-human-pose-estimation)
  
  []()
  
  []()
  
  []()
  
</details>

