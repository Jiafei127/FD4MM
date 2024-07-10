## **FD4MM**

Official PyTorch implementation for the paper:

> **Frequency Decoupling for Motion Magnification via Multi-Level Isomorphic Architecture**, ***CVPR 2024***.
>
> [Fei Wang](https://scholar.google.com.hk/citations?user=sdqv6pQAAAAJ&hl=zh-CN&oi=ao), [Dan Guo*](https://scholar.google.com.hk/citations?user=DsEONuMAAAAJ&hl=zh-CN&oi=ao), [Kun Li](https://scholar.google.com.hk/citations?user=UQ_bInoAAAAJ&hl=zh-CN&oi=ao), [Zhun Zhong](https://scholar.google.com.hk/citations?hl=zh-CN&user=nZizkQ0AAAAJ), [Meng Wang*](https://scholar.google.com.hk/citations?user=rHagaaIAAAAJ&hl=zh-CN&oi=ao).
>
> [![arXiv](https://img.shields.io/badge/arXiv-Paper-%3CCOLOR%3E.svg)](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Frequency_Decoupling_for_Motion_Magnification_via_Multi-Level_Isomorphic_Architecture_CVPR_2024_paper.pdf) |
> [![GitHub Stars](https://img.shields.io/github/stars/Jiafei127/FD4MM)](https://github.com/Jiafei127/FD4MM) |
> [![](https://img.shields.io/github/license/Jiafei127/FD4MM)](https://github.com/Jiafei127/FD4MM/blob/main/LICENSE) |
> <a href=' '><img src='https://img.shields.io/badge/Demo-Open in Colab-blue'></a>

<p align="center">
<img src="https://github.com/Jiafei127/FD4MM/blob/main/fig/poster.png" width="90%"/>
</p>

> ## ✒️:Abstract
Video Motion Magnification (VMM) aims to reveal subtle and imperceptible motion information of objects in the macroscopic world. Prior methods directly model the motion field from the Eulerian perspective by Representation Learning that separates shape and texture or Multi-domain Learning from phase fluctuations. Inspired by the frequency spectrum we observe that the low-frequency components with stable energy always possess spatial structure and less noise making them suitable for modeling the subtle motion field. To this end, we present FD4MM a new paradigm of Frequency Decoupling for Motion Magnification with a Multi-level Isomorphic Architecture to capture multi-level high-frequency details and a stable low-frequency structure (motion field) in video space. Since high-frequency details and subtle motions are susceptible to information degradation due to their inherent subtlety and unavoidable external interference from noise we carefully design Sparse High/Low-pass Filters to enhance the integrity of details and motion structures and a Sparse Frequency Mixer to promote seamless recoupling. Besides we innovatively design a contrastive regularization for this task to strengthen the model's ability to discriminate irrelevant features reducing undesired motion magnification. Extensive experiments on both Real-world and Synthetic Datasets show that our FD4MM outperforms SOTA methods. Meanwhile, FD4MM reduces FLOPs by 1.63x and boosts inference speed by 1.68x than the latest method.

--- 

> ## 📅: Data Preparation
- Please refer to the dataset configuration of [EulerMormer](https://github.com/VUT-HFUT/EulerMormer).

- For **train datasets** from [Oh et al. ECCV 2018](https://github.com/12dmodel/deep_motion_mag), see the official repository [here](https://drive.google.com/drive/folders/19K09QLouiV5N84wZiTPUMdoH9-UYqZrX?usp=sharing).

- For **Real-world datatsets**, we used three settings:
  - [Static Dataset](https://drive.google.com/drive/folders/1Bm3ItPLhRxRYp-dQ1vZLCYNPajKqxZ1a)
  - [Dynamic Dataset](https://drive.google.com/drive/folders/1t5u8Utvmu6gnxs90NLUIfmIX0_5D3WtK)

- Real-world videos (or any self-prepared videos) need to be configured via the following:
  - Check the settings of val_dir in **config.py** and modify it if necessary.
  - To convert the **Real-world video** into frames:
    `mkdir VIDEO_NAME && ffmpeg -i VIDEO_NAME.mp4 -f image2 VIDEO_NAME/%06d.png`
    
    eg, `mkdir ./val_baby && ffmpeg -i ./baby.avi -f image2 ./val_baby/%06d.png`
> Tips: ffmpeg can also be installed by conda.
  - Modify the frames into **frameA/frameB/frameC**:
    `python make_frameACB.py `(remember adapt the 'if' at the beginning of the program to select videos.)
> Tips: Thanks to a fellow friend [Peng Zheng](https://github.com/ZhengPeng7/motion_magnification_learning-based) for the help!



> ## 📑: Performance Comparison
> - For **Synthetic Test Dataset:** 
> ![test](https://github.com/Jiafei127/FD4MM/blob/main/fig/perf.png   'Performance Comparison')

> - For **Real-world Test Dataset:** 
> ![test](https://github.com/Jiafei127/FD4MM/blob/main/fig/sdperf.png   'sdPerformance Comparison')

> ## 🔖:Citation

If you found this code useful please consider citing our [FD4MM](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_Frequency_Decoupling_for_Motion_Magnification_via_Multi-Level_Isomorphic_Architecture_CVPR_2024_paper.html), [EulerMormer](https://arxiv.org/abs/2312.04152):
```
@inproceedings{wang2024frequency,
  title={Frequency decoupling for motion magnification via multi-level isomorphic architecture},
  author={Wang, Fei and Guo, Dan and Li, Kun and Zhong, Zhun and Wang, Meng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18984--18994},
  year={2024}
}
```
```
@inproceedings{wang2024eulermormer,
  title={Eulermormer: Robust eulerian motion magnification via dynamic filtering within transformer},
  author={Wang, Fei and Guo, Dan and Li, Kun and Wang, Meng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={6},
  pages={5345--5353},
  year={2024}
}
```
