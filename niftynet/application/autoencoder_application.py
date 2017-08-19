import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.io.image_reader import ImageReader
from niftynet.layer.pad import PadLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer

SUPPORTED_INPUT = {'image', 'feature'}
SUPPORTED_INFERENCE = {'encode', 'encode-decode',
                       'sample', 'linear_interpolation'}


class AutoencoderApplication(BaseApplication):
    def __init__(self, net_param, action_param, is_training):
        BaseApplication.__init__(self)
        tf.logging.info('starting autoencoder application')

        self.is_training = is_training

        self.net_param = net_param
        self.action_param = action_param

        self.data_param = None
        self.autoencoder_param = None

    def initialise_dataset_loader(self, data_param=None, task_param=None):
        self.data_param = data_param
        self.autoencoder_param = task_param

        # read each line of csv files into an instance of Subject
        if self.is_training:
            self.reader = ImageReader(SUPPORTED_INPUT)
        else:  # in the inference process use image input only
            self.reader = ImageReader(['image'])
        self.reader.initialise_reader(data_param, task_param)

        augmentation_layers = []
        if self.is_training:
            if self.action_param.random_flipping_axes != -1:
                augmentation_layers.append(RandomFlipLayer(
                    flip_axes=self.action_param.random_flipping_axes))
            if self.action_param.scaling_percentage:
                augmentation_layers.append(RandomSpatialScalingLayer(
                    min_percentage=self.action_param.scaling_percentage[0],
                    max_percentage=self.action_param.scaling_percentage[1]))
            if self.action_param.rotation_angle:
                augmentation_layers.append(RandomRotationLayer(
                    min_angle=self.action_param.rotation_angle[0],
                    max_angle=self.action_param.rotation_angle[1]))
        self.reader.add_preprocessing_layers(augmentation_layers)

    def initialise_sampler(self):
        """
        set samplers take self.reader as input and generates
        sequences of ImageWindow that will be fed to the networks
        This function sets self.sampler
        """
        raise NotImplementedError

    def initialise_network(self):
        """
        This function create an instance of network
        sets self.net
        :return: None
        """
        raise NotImplementedError

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        raise NotImplementedError

    def interpret_output(self, batch_output):
        raise NotImplementedError


class AutoencoderFactory(object):
    @staticmethod
    def create(name):
        if name == "vae":
            from niftynet.network.vae import VAE
            return VAE
        else:
            tf.logging.fatal("network: \"{}\" not implemented".format(name))
            raise NotImplementedError
