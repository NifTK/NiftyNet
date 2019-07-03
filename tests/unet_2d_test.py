from __future__ import absolute_import, print_function

import os
import unittest

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.unet_2d import UNet2D
from tests.niftynet_testcase import NiftyNetTestCase

@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true",
                 'Skipping slow tests')
class UNet3DTest(NiftyNetTestCase):
    def test_2d_shape(self):
        #input_shape = (2, 572, 572, 3)
        input_shape = (2, 180, 180, 3)
        x = tf.ones(input_shape)

        unet_instance = UNet2D(num_classes=2)
        out = unet_instance(x, is_training=True)
        print(unet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            #self.assertAllClose((2, 388, 388, 2), out.shape)
            self.assertAllClose((2, 4, 4, 2), out.shape)

    def test_2d_reg_shape(self):
        #input_shape = (2, 572, 572, 5)
        input_shape = (2, 180, 180, 5)
        x = tf.ones(input_shape)

        unet_instance = UNet2D(num_classes=2,
                               w_regularizer=regularizers.l2_regularizer(0.4))
        out = unet_instance(x, is_training=True)
        print(unet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            #self.assertAllClose((2, 388, 388, 2), out.shape)
            self.assertAllClose((2, 4, 4, 2), out.shape)


if __name__ == "__main__":
    tf.test.main()
