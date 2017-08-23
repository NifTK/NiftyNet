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

SUPPORTED_LOSS_GAN = {
    'CrossEntropy': 'niftynet.layer.loss_gan.cross_entropy',
}

SUPPORTED_LOSS_SEGMENTATION = {
    "CrossEntropy":
        'niftynet.layer.loss_segmentation.cross_entropy',
    "Dice":
        'niftynet.layer.loss_segmentation.dice',
    "Dice_NS":
        'niftynet.layer.loss_segmentation.dice_nosquare',
    "GDSC":
        'niftynet.layer.loss_segmentation.generalised_dice_loss',
    "WGDL":
        'niftynet.layer.loss_segmentation.wasserstein_generalised_dice_loss',
    "SensSpec":
        'niftynet.layer.loss_segmentation.sensitivity_specificity_loss',
    "L1Loss":
        'niftynet.layer.loss_segmentation.l1_loss',
    "L2Loss":
        'niftynet.layer.loss_segmentation.l2_loss',
    "Huber":
        'niftynet.layer.loss_segmentation.huber_loss'
}

SUPPORTED_LOSS_AUTOENCODER = {
    "VariationalLowerBound":
        'niftynet.layer.loss_autoencoder.variational_lower_bound',
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


class ModuleFactory(object):
    SUPPORTED = None
    type_str = 'object'

    @classmethod
    def create(cls, name):
        try:
            return select_module(name, cls.SUPPORTED)
        except ImportError:
            tf.logging.fatal(
                "{}: \"{}\" not implemented".format(cls.type_str, name))
            raise NotImplementedError


class ApplicationNetFactory(ModuleFactory):
    SUPPORTED = SUPPORTED_NETWORK
    type_str = 'network'


class ApplicationFactory(ModuleFactory):
    SUPPORTED = SUPPORTED_APP
    type_str = 'application'


class LossGANFactory(ModuleFactory):
    SUPPORTED = SUPPORTED_LOSS_GAN
    type_str = 'GAN loss'


class LossSegmentationFactory(ModuleFactory):
    SUPPORTED = SUPPORTED_LOSS_SEGMENTATION
    type_str = 'segmentation loss'


class LossAutoencoderFactory(ModuleFactory):
    SUPPORTED = SUPPORTED_LOSS_AUTOENCODER
    type_str = 'autoencoder loss'
