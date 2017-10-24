# Developing a new network architecture
NiftyNet allows users create new network, and choose to use the network by
specifying command line argument.  To fully utilise this feature, a customised
network should be prepared in the following steps:

###### Step 1.
   Create a new network file, e.g. `new_net.py` and place this inside a folder
   of your choice, e.g. `my_network_collection` together with a new `__init__.py`
   file.

###### Step 2.
   Make sure this folder can be discovered by NiftyNet by doing **either** of the
   following:
   * Place `my_network_collection` inside `$NIFTYNET_HOME/niftynetext/network`
   (with `$NIFTYNET_HOME` defined by the [NiftyNet global `home` setting][glob-conf]).
   * Append the location of this folder (i.e. **the folder where this folder is
   located**) to your `$PYTHONPATH`.

[glob-conf]: ../../config/README.md#global-niftynet-settings

###### Step 3.
   Create a new class, e.g. `NewNet` in `new_net.py` by inheriting the
   `BaseNet` class from `niftynet.network.base_net`.  `niftynet.network.toynet`
   is a minimal working example of a fully convolutional network, can be a
   starting point for `NewNet`.

###### Step 4.
   In the `NewNet` class, implement `__init__()` function for network property
   initialisations, and implement `layer_op()` for network connections.

   The network properties can be used to specify the number of channels, kernel
   dilation factors, as well as sub-network components of the network.

   An example of sub-networks composition is presented in
   [Simulator Gan](./simulator_gan.py).

   The layer operation function `layer_op()` should specify how the input
   tensors are connected to network layers.  For basic building blocks, using
   the ones in `niftynet/layer/` are recommended. as the layers are implemented
   in a modular design (convenient for parameter sharing) and can handle 2D,
   2.5D and 3D cases in a unified manner whenever possible.

###### Step 5.
   Finally training the network could be done by specifying the newly
   implemented network in the command line argument `--name my_network_collection.new_net.NewNet`

   (`my_network_collection.new_net` refer to the `new_net.py` file, and `NewNet`
   is the class name with in `new_net.py`)

   e.g., training new network with segmentation application using pip installed NiftyNet:
   ```
   net_segment train -c /path/to/customised_config \
                     --name my_network_collection.new_net.NewNet
   ```
   or using NiftyNet cloned from [GitHub](https://github.com/NifTK/NiftyNet) or
   [CMICLab](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet):
   ```
   python net_segment.py train -c /path/to/customised_config \
                               --name my_network_collection.new_net.NewNet
   ```

###### Step 6.
   Please consider submitting the design to our model zoo.


# Network model references
This section summarises the implemented network models in
[network](./).

All networks can be applied in 2D, 2.5D and 3D configurations and are
reimplemented from their original presentation with their default parameters.

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

