# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer


class GANImageBlock(TrainableLayer):
    def __init__(self,
                 generator,
                 discriminator,
                 clip=None,
                 name='GAN_image_block'):
        self._generator = generator
        self._discriminator = discriminator
        self.clip = clip
        super(GANImageBlock, self).__init__(name=name)

    def layer_op(self,
                 random_source,
                 training_image,
                 conditioning,
                 is_training):
        shape_to_generate = training_image.shape.as_list()[1:]
        fake_image = self._generator(random_source,
                                     shape_to_generate,
                                     conditioning,
                                     is_training)

        fake_logits = self._discriminator(fake_image,
                                          conditioning,
                                          is_training)
        if self.clip:
            with tf.name_scope('clip_real_images'):
                training_image = tf.maximum(
                    -self.clip,
                    tf.minimum(self.clip, training_image))
        real_logits = self._discriminator(training_image,
                                          conditioning,
                                          is_training)
        return fake_image, real_logits, fake_logits


class BaseGenerator(TrainableLayer):
    def __init__(self, name='generator', *args, **kwargs):
        super(BaseGenerator, self).__init__(name=name)


class BaseDiscriminator(TrainableLayer):
    def __init__(self, name='discriminator', *args, **kwargs):
        super(BaseDiscriminator, self).__init__(name=name)
