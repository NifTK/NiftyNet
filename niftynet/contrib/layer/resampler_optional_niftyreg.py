# Flag stating whether C++/CUDA image resampling is available
HAS_NIFTYREG_RESAMPLING = False

try:
    from niftyreg_image_resampling import NiftyregImageResamplingLayer
    import niftyreg_image_resampling as resampler_module

    ResamplerOptionalNiftyRegLayer = NiftyregImageResamplingLayer

    HAS_NIFTYREG_RESAMPLING = True
except ImportError:
    import tensorflow as tf

    tf.logging.warning('''
    niftyreg_image_resampling is not installed; falling back onto
    niftynet.layer.resampler.ResamplerLayer. To install
    niftyreg_image_resampling please see
    niftynet/contrib/niftyreg_image_resampling/README.md
    ''')

    from niftynet.layer.resampler import ResamplerLayer
    import niftynet.layer.resampler as resampler_module

    ResamplerOptionalNiftyRegLayer = ResamplerLayer


# Passthrough of supported boundary types
SUPPORTED_BOUNDARY = resampler_module.SUPPORTED_BOUNDARY


# Passthrough of supported interpolation types
SUPPORTED_INTERPOLATION = resampler_module.SUPPORTED_INTERPOLATION
