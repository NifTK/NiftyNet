# NiftyNet
<img src="https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/raw/master/niftynet-logo.png" width="263" height="155">

[![build status](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/badges/master/build.svg)](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/commits/master)
[![coverage report](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/badges/master/coverage.svg)](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/commits/master)

NiftyNet is a [TensorFlow][tf]-based open-source convolutional neural networks (CNN) platform for research in medical image analysis and computer-assisted intervention.
NiftyNet is a consortium of multiple research groups (WEISS -- [Wellcome EPSRC Centre for Interventional and Surgical Sciences][weiss], CMIC -- [Centre for Medical Image Computing][cmic], HIG -- High-dimensional Imaging Group), where WEISS acts as a consortium lead.
**NiftyNet is not intended for clinical use**.


### Features
* Easy-to-customise interfaces of network components
* Designed for sharing networks and pretrained models
* Designed to support 2-D, 2.5-D, 3-D, 4-D inputs*
* Efficient discriminative training with multiple-GPU support
* Implemented recent networks (HighRes3DNet, 3D U-net, V-net, DeepMedic)
* Comprehensive evaluation metrics for medical image segmentation

 <sup>*2.5-D: volumetric images processed as a stack of 2D slices;
4-D: co-registered multi-modal 3D volumes</sup>

### Getting started
Please follow the links for [demos](./demos) and [network (re-)implementations](./niftynet/network).

### Contributing
Feature requests and bug reports are collected on [Issues](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/issues).

Contributors are encouraged to take a look at [CONTRIBUTING.md](./CONTRIBUTING.md).

### Citing NiftyNet
If you use NiftyNet, please cite the following paper:
```
@InProceedings{niftynet17,
  author = {Li, Wenqi and Wang, Guotai and Fidon, Lucas and Ourselin, Sebastien and Cardoso, M. Jorge and Vercauteren, Tom},
  title = {On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task},
  booktitle = {International Conference on Information Processing in Medical Imaging (IPMI)},
  year = {2017}
}
```

### Licensing and Copyright

Copyright 2017 NiftyNet Contributors.
Released under the Apache License, Version 2.0. Please see the LICENSE file for details.

### Acknowledgements
This project is grateful for the support from the [Wellcome Trust][wt], the [Engineering and Physical Sciences Research Council (EPSRC)][epsrc], the [National Institute for Health Research (NIHR)][nihr], the [Department of Health (DoH)][doh], [University College London (UCL)][ucl], the [Science and Engineering South Consortium (SES)][ses], the [STFC Rutherford-Appleton Laboratory][ral], and [NVIDIA][nvidia].

[cmic]: http://cmic.cs.ucl.ac.uk
[ucl]: http://www.ucl.ac.uk
[tf]: https://www.tensorflow.org/
[weiss]: http://www.ucl.ac.uk/surgical-interventional-sciences
[wt]: https://wellcome.ac.uk/
[epsrc]: https://www.epsrc.ac.uk/
[nihr]: https://www.nihr.ac.uk/
[doh]: https://www.gov.uk/government/organisations/department-of-health
[ses]: https://www.ses.ac.uk/
[ral]: http://www.stfc.ac.uk/about-us/where-we-work/rutherford-appleton-laboratory/
[nvidia]: http://www.nvidia.com

