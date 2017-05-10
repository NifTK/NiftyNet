import tensorflow as tf

from base import Layer
from convolution import ConvolutionalLayer

class DeepMedic(Layer):
    def __init__(self,
                 num_classes):
        self.layer_name = 'DeepMedic'
        super(DeepMedic, self).__init__(name=self.layer_name)
        self.d_factor = 3 # downsampling factor
        self.crop_diff = (self.d_factor - 1) * 16
        self.conv_features = [30, 30, 40, 40, 40, 40, 50, 50]
        self.fc_features = [150, 150]
        self.acti_type = 'prelu'

    def layer_op(self, images, is_training, layer_id=-1):
        # image_size is defined as the largest context, then:
        #   downsampled path size: image_size / d_factor
        #   downsampled path output: image_size / d_factor - 16

        # to make sure same size of feature maps from both pathways:
        #   normal path size: (image_size / d_factor - 16) * d_factor + 16
        #   normal path output: (image_size / d_factor - 16) * d_factor

        # where 16 is fixed by the receptive field of conv layers
        # TODO: make sure label_size = image_size/d_factor - 16

        image_size = images.get_shape()[1]
        assert(images.get_shape()[2] == image_size)
        assert(images.get_shape()[3] == image_size)
        assert(image_size % self.d_factor == 0)
        assert(self.d_factor % 2 == 1) # to make the downsampling centered
        assert(image_size % 2 == 1) # to make the crop centered
        assert(image_size > self.d_factor * 16) # minimum receptive field

        return images
