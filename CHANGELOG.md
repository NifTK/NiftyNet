# Changelog
All notable changes to NiftyNet are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.0] - 2019-10-09
### Added
* isotropic random scaling option
* volume padding with user-specified constant
* subpixel layer for superresolution
* various loss functions for regression (smooth L1 loss, cosine loss etc.)
* handler for early stopping mechanism
* aggregator with multiple outputs including labels in CSV
* nnUNet, an improved version of UNet3D
* data augmentation with mixup and mixmatch
* documentation contents
* demo for learning rate scheduling
* demo for deep boosted regression
* initial integration of NiftyReg Resampler
* initial integration of CSV reader

### Fixed
* issue of loading binary values of NIfTI file
* various fixes in CI tests
* prefix name for aggregators
* various improvements in error messages
* issue of batch indices in the conditional random field
* issue of location selection in the weighted sampler
* model zoo: compatibility upgrade
* model zoo: new decathlon hippocampus dataset

### Changed
* feature normalisation types options: instance norm, group norm, batch norm
* convolution with padding option
* various documentation and docstrings
* defaulting to remove length one dimensions when saving a 5D volume

## [0.5.0] - 2019-02-04
### Added
* Version controlled model zoo with git-lfs
* Dice + entropy loss function
* Antialiasing when randomly scaling input images during training
* Support of multiple optimisers and gradients in applications

### Fixed
* An issue of rounding image sizes when `pixdim` is specified
* An issue of incorrect Dice when image patch does not include every class
* Numerous documentation issues

### Changed
* Tested with TensorFlow 1.12

## [0.4.0] - 2018-09-13
### Added
* `niftynet.layer`: new layers
    - Tversky loss function for image segmentation
    - Random affine augmentation layer
    - Random bias field augmentation layer
    - Group normalisation layer
    - Squeeze and excitation blocks
* Documentation
    - [Image reader and window sampler demo](https://github.com/NifTK/NiftyNet/tree/c457a5bb07284b030ce588d1d82b2907f7e4e65e/demos/module_examples)
    - [2D U-net demo](https://github.com/NifTK/NiftyNet/tree/c88cb1e4c6794ebbd6bf681901ee8902da41d7eb/demos/unet)
    - [Dense CRF layer demo](https://github.com/NifTK/NiftyNet/tree/8b9f3e40b6d0f3db61fab34d3b7afca43b93723d/demos/crf_as_rnn)
    - [FAQ list](https://github.com/NifTK/NiftyNet/wiki/NiftyNet-FAQ)
* Misc.
    - [Subject id from filename with regular expression](https://niftynet.readthedocs.io/en/dev/filename_matching.html)
    - Versioning with python-versioneer
    - Tested with TensorFlow 1.10

### Changed
* `niftynet.engine`: improved core functions
    - IO modules based on `tf.data.Dataset` (breaking changes)
    - Decoupled the engine and event handlers
* Migrated the code repository, model zoo, and [niftynet.io](http://niftynet.io) source code to 
[github.com/niftk](https://github.com/niftk).

## [0.3.0] - 2018-05-15
### Added
* Support for 2D image loading optionally using `skimage`, `pillow`, or `simpleitk`
* Image reader and sampler with `tf.data.Dataset`
* Class-balanced image window sampler
* Random deformation as data augmentation with SimpleITK
* Segmentation loss with dense labels (multi-channel binary labels)
* Experimental features:
   - learning-based registration
   - image classification
   - model evaluation
   - new engine design with observer pattern

### Deprecated
* Deprecating the [CmicLab][cmiclab] repository in favour of [GitHub][github]

[cmiclab]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet
[github]: https://github.com/NifTK/NiftyNet

## [0.2.2] - 2018-01-30
### Added
* Improvements for running validation iterations during training

### Fixed
* Bugs when running validation iterations during training
* Minor bugs in loss function modules, histogram standardisation, user parameter parsing

## [0.2.1] - 2017-12-14
### Added
* Support for custom network / application as external modules
* Unified workspace directory via global configuration functionalities
* Model zoo for network / data sharing
* Automatic training / validation / test sets splitting
* Validation iterations during training
* Regression application
* 2D / 3D resampler layer
* Versioning functionality for better issue tracking
* Academic paper release: ["NiftyNet: a deep-learning platform for medical imaging"](https://arxiv.org/abs/1709.03485)
* How-to guides and a new theme for [the API and examples documentation](http://niftynet.readthedocs.io/)

## [0.2.0] - 2017-09-08
### Added
* Support for unsupervised learning networks, including GANs and auto-encoders
* An application engine for managing low-level operations required by different types of high-level applications
* NiftyNet is now [available on the Python Package Index](https://pypi.org/project/NiftyNet): `pip install niftynet`
* NiftyNet website up and running: http://niftynet.io
* API reference published online: http://niftynet.rtfd.io/en/dev/py-modindex.html
* NiftyNet source code mirrored on GitHub: https://github.com/NifTK/NiftyNet
* 5 new network implementations:
   1. DenseVNet
   1. HolisticNet
   1. SimpleGAN
   1. SimulatorGAN
   1. VariationalAutoencoder (VAE)

### Fixed
* Bugs (30+ issues resolved)

## 0.1.1 - 2017-08-08
### Added
* Source code open sourced (CMICLab, GitHub)
* Initial PyPI package release
* Refactored sub-packages including `engine`, `application`, `layer`, `network`
* Command line entry points
* NiftyNet logo

### Fixed
* Bugs in data augmentation, I/O, sampler

[Unreleased]: https://github.com/NifTK/NiftyNet/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/NifTK/NiftyNet/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/NifTK/NiftyNet/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/NifTK/NiftyNet/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NifTK/NiftyNet/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/NifTK/NiftyNet/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/NifTK/NiftyNet/compare/v0.2.0.post1...v0.2.1
[0.2.0]: https://github.com/NifTK/NiftyNet/compare/v0.1.1...v0.2.0.post1
