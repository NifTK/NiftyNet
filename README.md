# NiftyNet
<img src="https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/raw/master/niftynet-logo.png" width="263" height="155">

[![build status](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/badges/master/build.svg)](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/commits/master)
[![coverage report](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/badges/master/coverage.svg)](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/commits/master)

NiftyNet is an open-source library for convolutional networks in medical image analysis.

NiftyNet was developed by the [Centre for Medical Image Computing][cmic] at
[University College London (UCL)][ucl].

### Features
* Easy-to-customise interfaces of network components
* Designed for sharing networks and pretrained models
* Designed to support 2-D, 2.5-D, 3-D, 4-D inputs*
* Efficient discriminative training with multiple-GPU support
* Implemented recent networks (HighRes3DNet, 3D U-net, V-net, DeepMedic)
* Comprehensive evaluation metrics for medical image segmentation

 <sup>*2.5-D: volumetric images processed as a stack of 2D slices;
4-D: co-registered multi-modal 3D volumes</sup>

### Usage
Please follow the links for [demos](./demos) and [network (re-)implementations](./niftynet/network).

### Contributing
Feature requests and bug reports are collected on [Issues](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/issues).

Contributors are encouraged to take a look at [CONTRIBUTING.md](./CONTRIBUTING.md).

### Citation
If you use this software, please cite:
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
This project was supported through an Innovative Engineering for Health award by
the Wellcome Trust and EPSRC (WT101957, NS/A000027/1), the National Institute
for Health Research University College London Hospitals Biomedical Research
Centre (NIHR BRC UCLH/UCL High Impact Initiative), UCL EPSRC CDT Scholarship
Award (EP/L016478/1), a UCL Overseas Research Scholarship, a UCL Graduate
Research Scholarship, and the Health Innovation Challenge Fund by the
Department of Health and Wellcome Trust (HICF-T4-275, WT 97914). The authors
would like to acknowledge that the work presented here made use of Emerald, a
GPU-accelerated High Performance Computer, made available by the Science &
Engineering South Consortium operated in partnership with the STFC
Rutherford-Appleton Laboratory.

[cmic]: http://cmic.cs.ucl.ac.uk
[ucl]: http://www.ucl.ac.uk

