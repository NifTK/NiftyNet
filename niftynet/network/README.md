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

## UNet
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
* Image size should be divisible by 8

## ScaleNet
Implementation of

Fidon, L., Li, W., Garcia-Peraza-Herrera, L.C., Ekanayake, J., Kitchen, N.,
Ourselin, S., Vercauteren, T. (2017). [Scalable convolutional networks for
brain tumour segmentation](https://arxiv.org/abs/1706.08124). In MICCAI 2017
##### Constraints
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

Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score for Imbalanced
Multi-class Segmentation using Holistic Convolutional Networks. MICCAI 2017
(BrainLes)

