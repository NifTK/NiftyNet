# Model zoo

Summary of the different implemented models, their constraints and default parameters
All networks can be applied in 2D, 2.5D and 3D configurations and are reimplemented from their original presentation with their default parameters

## UNet
Reimplementation of [the paper] (https://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf) by Cicek et al: [^1]
### Constraints
* Image size - 4 should be divisible by 8
* Label size should be more than 88
* border is 44

[^1]: Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016, October). 3d u-net: learning dense volumetric segmentation from sparse annotation. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 424-432). Springer International Publishing.

## VNet
Reimplementation of [the paper](http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf) by Milletari et al: [^2]

[^2]: Milletari, F., Navab, N., & Ahmadi, S. A. (2016, October). V-net: Fully convolutional neural networks for volumetric medical image segmentation. In 3D Vision (3DV), 2016 Fourth International Conference on (pp. 565-571). IEEE.
### Constraints
* Image size should be divisible by 8

## ScaleNet
Implementation of the paper by Fidon et al:[^3]
### Constraints
* More than one modality should be used  

[^3]:
## HighRes3dNet
Implementation of the paper by Li et al: [^4] 
### Constraints
* Image size should be divisible by 4  

[^4]: Li W, Wang G, Fidon L, Ourselin S, Cardoso M J, Vercauteren T, (2017)On the compactness, efficiency, and representation of 3D convolutional networks: Brain parcellation as a pretext task. In International Conference on Information Processing in Medical Imaging (IPMI)  


## DeepMedic
Reimplementation of [the paper](http://www.sciencedirect.com/science/article/pii/S1361841516301839) by Kamnistas et al: [^5]
### Constraints
* The downsampling factor (d_factor) should be odd
* Label size = [(image_size / d_ factor) - 16 ]*d_factor
* Image size should be divisible by d_factor

Example of appropriate configuration:
image_ size = 57, label_ size = 9, d_ factor = 3

[^5]: Kamnitsas, K., Ledig, C., Newcombe, V. F., Simpson, J. P., Kane, A. D., Menon, D. K., ... & Glocker, B. (2017). Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation. Medical Image Analysis, 36, 61-78.
## 