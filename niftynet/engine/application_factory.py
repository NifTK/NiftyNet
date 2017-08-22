import importlib

import tensorflow as tf

SUPPORTED_APP = {
    'net_segment':
        'niftynet.application.segmentation_application.SegmentationApplication',
    'net_autoencoder':
        'niftynet.application.autoencoder_application.AutoencoderApplication',
    'net_gan':
        'niftynet.application.gan_application.GANApplication'
}

SUPPORTED_NETWORK = {
    # GAN
    'simulator_gan':
        'niftynet.network.simulator_gan.SimulatorGAN',
    'simple_gan':
        'niftynet.network.simple_gan.SimpleGAN',

    # Segmentation
    "highres3dnet":
        'niftynet.network.highres3dnet.HighRes3DNet',
    "highres3dnet_small":
        'niftynet.network.highres3dnet_small.HighRes3DNetSmall',
    "highres3dnet_large":
        'niftynet.network.highres3dnet_large.HighRes3DNetLarge',
    "toynet":
        'niftynet.network.toynet.ToyNet',
    "unet":
        'niftynet.network.unet.UNet3D',
    "vnet":
        'niftynet.network.vnet.VNet',
    "dense_vnet":
        'niftynet.network.dense_vnet.DenseVNet',
    "deepmedic":
        'niftynet.network.deepmedic.DeepMedic',
    "scalenet":
        'niftynet.network.scalenet.ScaleNet',
    "holistic_scalenet":
        'niftynet.network.holistic_scalenet.HolisticScaleNet',

    # autoencoder
    "vae": 'niftynet.network.vae.VAE'
}


def select_module(module_name, lookup_table):
    if module_name in lookup_table:
        module_name = lookup_table.get(module_name, None)
    try:
        module_name, class_name = module_name.rsplit('.', 1)
    except ValueError:
        tf.logging.fatal('incorrect module name format {}'.format(module_name))
        raise ValueError

    try:
        the_module = getattr(importlib.import_module(module_name), class_name)
        return the_module
    except ImportError:
        raise ImportError


class ApplicationNetFactory(object):
    @staticmethod
    def create(name):
        try:
            return select_module(name, SUPPORTED_NETWORK)
        except ImportError:
            tf.logging.fatal("network: \"{}\" not implemented".format(name))
            raise NotImplementedError


class ApplicationFactory(object):
    @staticmethod
    def create(name):
        try:
            return select_module(name, SUPPORTED_APP)
        except ImportError:
            tf.logging.fatal("application: \"{}\" not implemented".format(name))
            raise NotImplementedError
