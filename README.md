# Status update - 2020-04-21

⚠️ **NiftyNet is not actively maintained anymore**. We have learned a lot in our journey and decided to redirect most of the development efforts towards [MONAI](https://github.com/Project-MONAI/MONAI/).

# NiftyNet

<img src="https://github.com/NifTK/NiftyNet/raw/dev/niftynet-logo.png" width="263" height="155">

[![pipeline status](https://gitlab.com/NifTK/NiftyNet/badges/dev/pipeline.svg)](https://github.com/NifTK/NiftyNet/commits/dev)
[![coverage report](https://gitlab.com/NifTK/NiftyNet/badges/dev/coverage.svg)](https://github.com/NifTK/NiftyNet)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/NifTK/NiftyNet/blob/dev/LICENSE)
[![PyPI version](https://badge.fury.io/py/NiftyNet.svg)](https://badge.fury.io/py/NiftyNet)

NiftyNet is a [TensorFlow][tf]-based open-source convolutional neural networks (CNN) platform for research in medical image analysis and image-guided therapy.
NiftyNet's modular structure is designed for sharing networks and pre-trained models.
Using this modular structure you can:

* Get started with established pre-trained networks using built-in tools
* Adapt existing networks to your imaging data
* Quickly build new solutions to your own image analysis problems

NiftyNet is a consortium of research organisations
(BMEIS -- [School of Biomedical Engineering and Imaging Sciences, King's College London][bmeis];
WEISS -- [Wellcome EPSRC Centre for Interventional and Surgical Sciences, UCL][weiss];
CMIC -- [Centre for Medical Image Computing, UCL][cmic];
HIG -- High-dimensional Imaging Group, UCL), where BMEIS acts as the consortium lead.


### Features

* Easy-to-customise interfaces of network components
* Sharing networks and pretrained models
* Support for 2-D, 2.5-D, 3-D, 4-D inputs*
* Efficient training with multiple-GPU support
* Implementation of recent networks (HighRes3DNet, 3D U-net, V-net, DeepMedic)
* Comprehensive evaluation metrics for medical image segmentation

 <sup>NiftyNet is not intended for clinical use.</sup>

 <sup>NiftyNet release notes are available [here][changelog].</sup>

 <sup>*2.5-D: volumetric images processed as a stack of 2D slices;
4-D: co-registered multi-modal 3D volumes</sup>

[changelog]: CHANGELOG.md


### Installation

1. Please install the appropriate [TensorFlow][tf] package*:
   * [`pip install "tensorflow==1.15.*"`][tf-pypi]
1. [`pip install niftynet`](https://pypi.org/project/NiftyNet/)

 <sup>All other NiftyNet dependencies are installed automatically as part of the pip installation process.

To install from the source repository, please checkout [the instructions](http://niftynet.readthedocs.io/en/dev/installation.html).</sup>

[tf-pypi-gpu]: https://pypi.org/project/tensorflow-gpu/
[tf-pypi]: https://pypi.org/project/tensorflow/


### Documentation
The API reference and how-to guides are available on [Read the Docs][rtd-niftynet].

[rtd-niftynet]: http://niftynet.rtfd.io/

### Useful links

* [NiftyNet website][niftynet-io]
* [NiftyNet source code on GitHub][niftynet-github]
* [NiftyNet Model zoo repository][niftynet-zoo]
* [NiftyNet Google Group / Mailing List][ml-niftynet]
* [Stack Overflow](https://stackoverflow.com/questions/tagged/niftynet) for general questions

[niftynet-io]: http://niftynet.io/
[niftynet-github]: https://github.com/NifTK/NiftyNet
[niftynet-zoo]: https://github.com/NifTK/NiftyNetModelZoo/blob/master/README.md
[ml-niftynet]: https://groups.google.com/forum/#!forum/niftynet


### Citing NiftyNet

If you use NiftyNet in your work, please cite [Gibson and Li, et al. 2018][cmpb2018]:

* E. Gibson\*, W. Li\*, C. Sudre, L. Fidon, D. I. Shakir, G. Wang, Z. Eaton-Rosen, R. Gray, T. Doel, Y. Hu, T. Whyntie, P. Nachev, M. Modat, D. C. Barratt, S. Ourselin, M. J. Cardoso† and T. Vercauteren† (2018)
[NiftyNet: a deep-learning platform for medical imaging][cmpb2018], _Computer Methods and Programs in Biomedicine_.
DOI: [10.1016/j.cmpb.2018.01.025][cmpb2018]


BibTeX entry:

```
@article{Gibson2018,
  title = "NiftyNet: a deep-learning platform for medical imaging",
  journal = "Computer Methods and Programs in Biomedicine",
  year = "2018",
  issn = "0169-2607",
  doi = "https://doi.org/10.1016/j.cmpb.2018.01.025",
  url = "https://www.sciencedirect.com/science/article/pii/S0169260717311823",
  author = "Eli Gibson and Wenqi Li and Carole Sudre and Lucas Fidon and
            Dzhoshkun I. Shakir and Guotai Wang and Zach Eaton-Rosen and
            Robert Gray and Tom Doel and Yipeng Hu and Tom Whyntie and
            Parashkev Nachev and Marc Modat and Dean C. Barratt and
            Sébastien Ourselin and M. Jorge Cardoso and Tom Vercauteren",
}
```
The NiftyNet platform originated in software developed for [Li, et al. 2017][ipmi2017]:

* Li W., Wang G., Fidon L., Ourselin S., Cardoso M.J., Vercauteren T. (2017)
[On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task.][ipmi2017]
In: Niethammer M. et al. (eds) Information Processing in Medical Imaging. IPMI 2017.
Lecture Notes in Computer Science, vol 10265. Springer, Cham.
DOI: [10.1007/978-3-319-59050-9_28][ipmi2017]


[ipmi2017]: https://doi.org/10.1007/978-3-319-59050-9_28
[cmpb2018]: https://doi.org/10.1016/j.cmpb.2018.01.025


### Licensing and Copyright

NiftyNet is released under [the Apache License, Version 2.0](https://github.com/NifTK/NiftyNet/blob/dev/LICENSE).

Copyright 2018 the NiftyNet Consortium.

### Acknowledgements

This project is grateful for the support from
the [Wellcome Trust][wt],
the [Engineering and Physical Sciences Research Council (EPSRC)][epsrc],
the [National Institute for Health Research (NIHR)][nihr],
the [Department of Health (DoH)][doh],
[Cancer Research UK][cruk],
[King's College London (KCL)][kcl],
[University College London (UCL)][ucl],
the [Science and Engineering South Consortium (SES)][ses],
the [STFC Rutherford-Appleton Laboratory][ral], and [NVIDIA][nvidia].

[bmeis]: https://www.kcl.ac.uk/lsm/research/divisions/imaging/index.aspx
[cmic]: http://cmic.cs.ucl.ac.uk
[ucl]: http://www.ucl.ac.uk
[kcl]: http://www.kcl.ac.uk
[cruk]: https://www.cancerresearchuk.org
[tf]: https://www.tensorflow.org/
[weiss]: http://www.ucl.ac.uk/weiss
[wt]: https://wellcome.ac.uk/
[epsrc]: https://www.epsrc.ac.uk/
[nihr]: https://www.nihr.ac.uk/
[doh]: https://www.gov.uk/government/organisations/department-of-health
[ses]: https://www.ses.ac.uk/
[ral]: http://www.stfc.ac.uk/about-us/where-we-work/rutherford-appleton-laboratory/
[nvidia]: http://www.nvidia.com

