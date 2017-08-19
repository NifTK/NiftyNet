# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer


# import niftynet.engine.logging as logging

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
        shape_to_generate = training_image.get_shape().as_list()[1:]
        fake_image = self._generator(
            random_source, shape_to_generate, conditioning, is_training)
        fake_logits = self._discriminator(
            fake_image, conditioning, is_training)
        if self.clip:
            with tf.name_scope('clip_real_images'):
                training_image = tf.maximum(
                    -self.clip,
                    tf.minimum(self.clip, training_image))
        real_logits = self._discriminator(
            training_image, conditioning, is_training)

        # with tf.name_scope('summaries_images'):
        # if len(fake_image.get_shape()) - 2 == 3:
        #    logging.image3_axial('fake', (fake_image / 2 + 1) * 127, 2, [logging.LOG])
        #    logging.image3_axial('real', tf.maximum(0., tf.minimum(255., (training_image / 2 + 1) * 127)), 2, [logging.LOG])
        # if len(fake_image.get_shape()) - 2 == 2:
        #    tf.summary.fake_image('fake', (fake_image / 2 + 1) * 127, 2, [logging.LOG])
        #    tf.summary.fake_image('real', tf.maximum(0., tf.minimum(255., (training_image / 2 + 1) * 127)), 2, [logging.LOG])
        return fake_image, real_logits, fake_logits


class BaseGenerator(TrainableLayer):
    def __init__(self, name='generator', *args, **kwargs):
        super(BaseGenerator, self).__init__(name=name)


class BaseDiscriminator(TrainableLayer):
    def __init__(self, name='discriminator', *args, **kwargs):
        super(BaseDiscriminator, self).__init__(name=name)
