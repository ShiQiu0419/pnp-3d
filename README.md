# PnP-3D 
This repository is for PnP-3D introduced in the following [paper](https://arxiv.org/abs/2108.07378):
 
"PnP-3D: A Plug-and-Play for 3D Point Clouds"  
[Shi Qiu](https://shiqiu0419.github.io/), [Saeed Anwar](https://saeed-anwar.github.io/), [Nick Barnes](http://users.cecs.anu.edu.au/~nmb/)  
IEEE Transactions on Pattern Analysis and Machine Intelligence (**TPAMI**), 2021

## Updates
* **18/08/2021** The paper is currently under review, and the codes will be released in the future. 
* **16/12/2021** The paper has been accepted in **TPAMI**. 
* **23/12/2021** The paper has been available on [IEEE Xplore](https://ieeexplore.ieee.org/document/9661313). 
* **07/01/2022** Sample codes (in both **pytorch** and **tensorflow**) are released. 

## Abstract
With the help of the deep learning paradigm, many point cloud networks have been invented for visual analysis. However, there is great potential for development of these networks since the given information of point cloud data has not been fully exploited. To improve the effectiveness of existing networks in analyzing point cloud data, we propose a plug-and-play module, PnP-3D, aiming to refine the fundamental point cloud feature representations by involving more local context and global bilinear response from explicit 3D space and implicit feature space. To thoroughly evaluate our approach, we conduct experiments on three standard point cloud analysis tasks, including classification, semantic segmentation, and object detection, where we select three state-of-the-art networks from each task for evaluation. Serving as a plug-and-play module, PnP-3D can significantly boost the performances of established networks. In addition to achieving state-of-the-art results on four widely used point cloud benchmarks, we present comprehensive ablation studies and visualizations to demonstrate our approach's advantages. The code will be available at https://github.com/ShiQiu0419/pnp-3d.

## Overview
<p align="center">
  <img width="1200" src="https://github.com/ShiQiu0419/pnp-3d/blob/main/feature_refine-1.png">
</p> 

## Visualization
<p align="center">
  <img width="1200" src="https://github.com/ShiQiu0419/pnp-3d/blob/main/vis1-1.png">
</p> 

## Paper and Citation
The paper can be downloaded from [here (arXiv)](https://arxiv.org/abs/2108.07378) and [here (ieee)](https://ieeexplore.ieee.org/document/9661313).  
If you find our paper/codes/results are useful, please cite:

    @article{qiu2021pnp-3d,
      title={PnP-3D: A Plug-and-Play for 3D Point Clouds},
      author={Qiu, Shi and Anwar, Saeed and Barnes, Nick},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2021},
      doi={10.1109/TPAMI.2021.3137794}
    }
