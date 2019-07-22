# NiftyNet GPU Image Resampling Module

## Purpose and Scope

This module provides a faster implementation of image resampling. For most usage scenarios, it is a drop-in replacement for niftynet.layer.resampler.ResamplerLayer, however, its feature set is limited to:

* ZERO (zero-padding), REPLICATE (clamping of intensities at edges), and SYMMETRIC (mirroring) boundaries
* NEAREST (constant), LINEAR, and BSPLINE (cubic spline) interpolation.
* Differentiation with respect to the floating image is a CPU-only operation.

To provide compatibility where this module is not installed, the following module can be used: niftynet.contrib.layer.resampler_optional_niftyreg.ResamplerOptionalNiftyRegLayer. This module will try to load NiftyregImageResamplingLayer, if that fails, it defaults to niftynet.layer.resampler.ResamplerLayer.

## Building and Installing

Building and installing is performed as usual via the setup file.

Building the module requires that a CUDA toolkit and CMake be installed, and nvcc and cmake can be found on the executables search path.
CMake variables can be overriden through the `override` command and `--settings`/`-s` switch as a list of colon (':') separated variable-value pairs. E.g., `python setup.py override -s "CMAKE_CXX_COMPILER:/usr/bin/g++-6:CMAKE_C_COMPILER:/usr/bin/gcc-6" build`.

Building was only tested with CUDA 9.0/gcc 5.4 and gcc 6.4, on Ubuntu 16.04 and 18.04.

## Acknowledgements

The image resampling code contained in this module is heavily based on code extracted from NiftyReg (https://sourceforge.net/projects/niftyreg/).

