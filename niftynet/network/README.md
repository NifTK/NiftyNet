# Model zoo

This page summarises the implemented network models in [network](./niftynet/network).

All networks can be applied in 2D, 2.5D and 3D configurations and are reimplemented from their original presentation with their default parameters.

## UNet
Reimplementation of 

Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., and Ronneberger, O. (2016). [3D U-net: Learning dense volumetric segmentation from sparse annotation](https://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf), In MICCAI 2016
##### Constraints
* Image size - 4 should be divisible by 8
* Label size should be more than 88
* border is 44



## VNet
Reimplementation of

Milletari, F., Navab, N., & Ahmadi, S. A. (2016). [V-net: Fully convolutional neural networks for volumetric medical image segmentation](http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf), In 3DV 2016 
##### Constraints
* Image size should be divisible by 8

## ScaleNet
Implementation of

Fidon, L., Li, W., Garcia-Peraza-Herrera, L.C., Ekanayake, J., Kitchen, N., Ourselin, S., Vercauteren, T. (2017). [Scalable convolutional networks for brain tumour segmentation](https://arxiv.org/abs/1706.08124). In MICCAI 2017
##### Constraints
* More than one modality should be used  


## HighRes3dNet
Implementation of 

Li W, Wang G, Fidon L, Ourselin S, Cardoso M J, Vercauteren T, (2017). [On the compactness, efficiency, and representation of 3D convolutional networks: Brain parcellation as a pretext task](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_28), In IPMI 2017
##### Constraints
* Image size should be divisible by 8    


## DeepMedic
Reimplementation of

Kamnitsas, K., Ledig, C., Newcombe, V. F., Simpson, J. P., Kane, A. D., Menon, D. K., Rueckert, D., Glocker, B. (2017). [Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation](http://www.sciencedirect.com/science/article/pii/S1361841516301839), MedIA 36, 61-78
##### Constraints
* The downsampling factor (d_factor) should be odd
* Label size = [(image_size / d_ factor) - 16 ]*d_factor
* Image size should be divisible by d_factor

Example of appropriate configuration:
image_ size = 57, label_ size = 9, d_ factor = 3



## To develop a new network architecture
1. Create a `niftynet/network/new_net.py` inheriting `BaseNet` from `niftynet.layer.base_net`
1. Implement `layer_op()` function using the building blocks in `niftynet/layer/` or creating new layers
1. Import `niftynet.network.new_net` to the `NetFactory` class in `niftynet/__init__.py`
1. Train the network with `python net_segmentation.py train -c /path/to/customised_config`
