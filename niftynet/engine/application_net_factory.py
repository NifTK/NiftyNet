import importlib

SHORTHANDS = {
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
    "vae":'niftynet.network.vae.VAE'
}


class ApplicationNetFactory(object):
    @staticmethod
    def create(name):
        if name in SHORTHANDS:
            name = SHORTHANDS.get(name, None)
        try:
            net_module_name, cls = name.rsplit('.', 1)
            net_class = getattr(importlib.import_module(net_module_name), cls)
            return net_class
        except ImportError as e:
            tf.logging.fatal("network: \"{}\" not implemented".format(name))
            raise NotImplementedError
