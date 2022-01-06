# PnP-3D 
This is a pytorch implementation of PnP-3D module.

## Note
* Although we use [knn](https://github.com/ShiQiu0419/pnp-3d/blob/4e516ed750d0764176cd6f50dff4194f0905607f/pytorch/pnp3d.py#L37) in these sample codes, you may simply replace it with other searching algorithm/neighbor indices.
* It's better to place PnP-3D module behind the encoder in your network, but an optimal value of *k* may vary case by case. 

## Usage
* Classification: [DGCNN](https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py), [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_cls_ssg.py), etc.
* Segmentation: [PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_sem_seg.py), [CloserLook3D](https://github.com/zeliu98/CloserLook3D/blob/master/pytorch/models/backbones/resnet.py)
* Detection [ImVoteNet](https://github.com/facebookresearch/imvotenet/blob/main/models/backbone_module.py)

Open an issue if you have question about any experimental setting reported in our paper.  
