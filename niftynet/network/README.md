# Developing a new network architecture
NiftyNet allows users create new network, and choose to use the network by
specifying command line argument.

_To get started, check out [the developer's guides](https://niftynet.readthedocs.io/en/dev/index.html#guides)._


# Network model references
This section summarises the implemented network models in
[network](./).

All networks can be applied in 2D, 2.5D and 3D configurations and are
reimplemented from their original presentation with their default parameters.

_See also: [NiftyNet model zoo](https://github.com/NifTK/NiftyNetModelZoo)._


## UNet 2D
Reimplementation of

Ronneberger, O., Fischer, P., and Brox, T. (2015). 
[U-Net: Convolutional Networks for Buomedical Image 
Segmentation](https://arxiv.org/pdf/1505.04597.pdf)
In MICCAI 2015
##### Constraints
* input should be 2D


## UNet 3D
Reimplementation of

Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., and Ronneberger, O.
(2016). [3D U-net: Learning dense volumetric segmentation from sparse
annotation](https://lmb.informatik.uni-freiburg.de/Publications/2016/CABR16/cicek16miccai.pdf),
In MICCAI 2016
##### Constraints
* Image size - 4 should be divisible by 8
* Label size should be more than 88
* border is 44


## VNet
Reimplementation of

Milletari, F., Navab, N., & Ahmadi, S. A. (2016). [V-net: Fully convolutional
neural networks for volumetric medical image
segmentation](http://campar.in.tum.de/pub/milletari2016Vnet/milletari2016Vnet.pdf),
In 3DV 2016
##### Constraints
* Input size should be divisible by 8
* Input should be either 2D or 3D


## DenseVNet
Reimplementation of

Gibson, E., Giganti, F., Hu, Y., Bonmati, E., Bandula, S., Gurusamy, K., Davidson, B.,
Pereira, S.P., Clarkson, M.J. & Barratt, D.C. (2018). [Automatic Multi-organ Segmentation 
on Abdominal CT with Dense V-networks](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6076994/),
in TMI 2018
##### Constraints
* Input size should be divisible by 2*dilation_rates


## ScaleNet
Implementation of

Fidon, L., Li, W., Garcia-Peraza-Herrera, L.C., Ekanayake, J., Kitchen, N.,
Ourselin, S., Vercauteren, T. (2017). [Scalable convolutional networks for
brain tumour segmentation](https://arxiv.org/abs/1706.08124). In MICCAI 2017
##### Constraints
* Image size should be divisible by 8
* More than one modality should be used


## HighRes3dNet
Implementation of

Li W, Wang G, Fidon L, Ourselin S, Cardoso M J, Vercauteren T, (2017). [On the
compactness, efficiency, and representation of 3D convolutional networks: Brain
parcellation as a pretext
task](https://link.springer.com/chapter/10.1007/978-3-319-59050-9_28), In IPMI
2017
##### Constraints
* Image size should be divisible by 8
##### Versions
* default
* large (additional 3x3x3 convolution)
* small (initial stride-2 convolution)


## DeepMedic
Reimplementation of

Kamnitsas, K., Ledig, C., Newcombe, V. F., Simpson, J. P., Kane, A. D., Menon,
D. K., Rueckert, D., Glocker, B. (2017). [Efficient multi-scale 3D CNN with
fully connected CRF for accurate brain lesion
segmentation](http://www.sciencedirect.com/science/article/pii/S1361841516301839),
MedIA 36, 61-78
##### Constraints
* The downsampling factor (`d_factor`) should be odd
* Label size = [(image_size / d_ factor) - 16 ]*d_factor
* Image size should be divisible by d_factor

Example of appropriate configuration for training:
image spatial window size = 57, label spatial window size = 9, d_ factor = 3

and for inference:
image spatial window size = 105, label spatial window size = 57, d_ factor = 3


## HolisticNet
Implementation of

Fidon, L. et. al. (2017) [Generalised Wasserstein Dice Score for Imbalanced
Multi-class Segmentation using Holistic Convolutional Networks],
(https://arxiv.org/abs/1707.00478), MICCAI 2017
(BrainLes)
##### Constraints
* Image size should be divisible by 8


## InterventionalAffineNet (INetAffine)
Implementation of the affine registration network presented in:

Hu, Y., Modat, M., Gibson, E., Ghavami, N., Bonmati, E., Moore, C.M.,
Emberton, M., Noble, J.A., Barratt, D.C. & Vercauteren, T. (2017)
[Label-driven weakly-supervised learning for multimodal deformable image 
registration](https://arxiv.org/abs/1711.01666). In ISBI 2018

Hu, Y., Modat, M., Gibson, E., Li, W., Ghavami, N., Bonmati, E., Wang, G., 
Bandula, S., Moore, C.M., Emberton, Ourselin, S., M., Noble, J.A., 
Barratt, D.C. & Vercauteren, T. (2018) [Weakly-Supervised Convolutional 
Neural Networks for Multimodal Image Registration]
(https://arxiv.org/abs/1807.03361). In MedIA 2018
##### Constraints
* Only 2D or 3D input images supported


## InterventionalDenseNet (INetDense)
Implementation of the dense registration network presented in:

Hu, Y., Modat, M., Gibson, E., Ghavami, N., Bonmati, E., Moore, C.M.,
Emberton, M., Noble, J.A., Barratt, D.C. & Vercauteren, T. (2017)
[Label-driven weakly-supervised learning for multimodal deformable image 
registration](https://arxiv.org/abs/1711.01666). In ISBI 2018

Hu, Y., Modat, M., Gibson, E., Li, W., Ghavami, N., Bonmati, E., Wang, G., 
Bandula, S., Moore, C.M., Emberton, Ourselin, S., M., Noble, J.A., 
Barratt, D.C. & Vercauteren, T. (2018) [Weakly-Supervised Convolutional 
Neural Networks for Multimodal Image Registration]
(https://arxiv.org/abs/1807.03361). In MedIA 2018
##### Constraints
- input spatial rank should be either 2 or 3 (2D or 3D images only)
* fixed image size should be divisible by 16


## InterventionalHybrid (INetHybridPreWarp)
Implementation of the hybrid affine-dense registration network presented in:

Hu, Y., Modat, M., Gibson, E., Ghavami, N., Bonmati, E., Moore, C.M.,
Emberton, M., Noble, J.A., Barratt, D.C. & Vercauteren, T. (2017)
[Label-driven weakly-supervised learning for multimodal deformable image 
registration](https://arxiv.org/abs/1711.01666). In ISBI 2018

Hu, Y., Modat, M., Gibson, E., Li, W., Ghavami, N., Bonmati, E., Wang, G., 
Bandula, S., Moore, C.M., Emberton, Ourselin, S., M., Noble, J.A., 
Barratt, D.C. & Vercauteren, T. (2018) [Weakly-Supervised Convolutional Neural Networks 
for Multimodal Image Registration](https://arxiv.org/abs/1807.03361). In MedIA 2018

##### Constraints
- input spatial rank should be either 2 or 3 (2D or 3D images only)
* fixed image size should be divisible by 16


## ResNet
Reimplementation of

He, K., Zhang, X., Ren, S., Sun, J. (2016) [Identity Mappings in Deep 
Residual Networks](https://arxiv.org/abs/1603.05027). In ECCV 2016
  

## Squeeze-and-Excitation Net (SE_ResNet)
3D reimplementation of 

Hu, J., Shen, L., Albanie, S., Sun, G. & Wu, E. (2018)
[Squeeze-and-Excitation Networks](arXiv:1709.01507v2), 
CVPR 2018


## GenericGAN (simple_gan)
Demo for basic implementation of a generative adversarial network

See for instance:
Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D.,
Ozair, S., Courville, A. & Bengio, Y. (2014)[Generative Adversarial
Nets](http://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf)
In NIPS 2014


## SimulatorGAN
Reimplementation of

Hu, Y., Gibson, E., Lee, L., Xie, W., Barratt, D.C., Vercauteren, T., Noble, A. (2017), 
[Freehand Ultrasound Image Simulation with Spatially-Conditioned 
Generative Adversarial Networks](https://arxiv.org/abs/1707.05392),
In: MICCAI RAMBO 2017


## ToyNet
Basic two-layer convolutional neural network for simple testing. 


## VAE
Implementation of a variational autoencoder network based on 

Kingma, D.P. & Welling, M. (2014)[Auto-Encoding Variational 
Bayes](https://arxiv.org/abs/1312.6114)






