# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf
#import niftynet.engine.logging as logging

from niftynet.layer.base_layer import TrainableLayer

class GANImageBlock(TrainableLayer):
    def __init__(self, generator, discriminator, name='GAN_image_block'):
        self._generator = generator
        self._discriminator = discriminator
        super(GANImageBlock, self).__init__(name=name)

    def layer_op(self, random_source, population, conditioning, is_training):
        image = self._generator(random_source, population.get_shape().as_list()[1:], conditioning, is_training)
        fake_logits = self._discriminator(image, conditioning, is_training)
        real_logits = self._discriminator(tf.maximum(-2., tf.minimum(2., population)), conditioning, is_training)
        #if len(image.get_shape()) - 2 == 3:
        #    logging.image3_axial('fake', (image / 2 + 1) * 127, 1, [logging.LOG])
        #    logging.image3_axial('real', tf.maximum(0., tf.minimum(255., (population / 2 + 1) * 127)), 1, [logging.LOG])
        #if len(image.get_shape()) - 2 == 2:
        #    tf.summary.image('fake', (image / 2 + 1) * 127, 1, [logging.LOG])
        #    tf.summary.image('real', tf.maximum(0., tf.minimum(255., (population / 2 + 1) * 127)), 1, [logging.LOG])
        return image, real_logits, fake_logits

class BaseGenerator(TrainableLayer):
    def __init__(self, name='generator', *args, **kwargs):
        super(BaseGenerator, self).__init__(name='generator')

class BaseDiscriminator(TrainableLayer):
    def __init__(self, name='discriminator', *args, **kwargs):
        super(BaseDiscriminator, self).__init__(name='discriminator')
