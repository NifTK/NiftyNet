from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.vnet import VNet
from tests.niftynet_testcase import NiftyNetTestCase

class VNetTest(NiftyNetTestCase):
    def test_3d_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        x = tf.ones(input_shape)

        # vnet_instance = VNet(num_classes=160)
        vnet_instance = VNet(num_classes=160)
        out = vnet_instance(x, is_training=True)
        print(vnet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 32, 160), out.shape)

    def test_2d_shape(self):
        input_shape = (2, 32, 32, 1)
        x = tf.ones(input_shape)

        # vnet_instance = VNet(num_classes=160)
        vnet_instance = VNet(num_classes=160)
        out = vnet_instance(x, is_training=True)
        print(vnet_instance.num_trainable_params())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 160), out.shape)

    def test_3d_reg_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        x = tf.ones(input_shape)

        # vnet_instance = VNet(num_classes=160)
        vnet_instance = VNet(
            num_classes=160,
            w_regularizer=regularizers.l2_regularizer(0.4),
            b_regularizer=regularizers.l2_regularizer(0.4))
        out = vnet_instance(x, is_training=True)
        print(vnet_instance.num_trainable_params())
        # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 32, 160), out.shape)

    def test_2d_reg_shape(self):
        input_shape = (2, 32, 32, 1)
        x = tf.ones(input_shape)

        # vnet_instance = VNet(num_classes=160)
        vnet_instance = VNet(
            num_classes=160,
            w_regularizer=regularizers.l2_regularizer(0.4),
            b_regularizer=regularizers.l2_regularizer(0.4))
        out = vnet_instance(x, is_training=True)
        print(vnet_instance.num_trainable_params())
        # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # print(vnet_instance.regularizer_loss())

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 160), out.shape)


if __name__ == "__main__":
    tf.test.main()
