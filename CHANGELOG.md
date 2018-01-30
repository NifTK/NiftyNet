# Changelog
All notable changes to NiftyNet are documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/NifTK/NiftyNet/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/NifTK/NiftyNet/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/NifTK/NiftyNet/compare/v0.2.0.post1...v0.2.1
[0.2.0]: https://github.com/NifTK/NiftyNet/compare/v0.1.1...v0.2.0.post1
