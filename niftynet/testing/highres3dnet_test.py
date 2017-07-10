from __future__ import absolute_import, print_function
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.network.highres3dnet import HighRes3DNet


class HighRes3DNetTest(tf.test.TestCase):
    def test_2d_shape(self):
        input_shape = (2, 32, 32, 1)
        x = tf.ones(input_shape)

        highres_layer = HighRes3DNet(num_classes=5)
        out = highres_layer(x, is_training=True)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 5), out.shape)

    def test_2d_reg_shape(self):
        input_shape = (2, 32, 32, 1)
        x = tf.ones(input_shape)

        highres_layer = HighRes3DNet(
            num_classes=5,
            w_regularizer=regularizers.l2_regularizer(0.5),
            b_regularizer=regularizers.l2_regularizer(0.5))
        out = highres_layer(x, is_training=True)
        # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 5), out.shape)

    def test_3d_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        x = tf.ones(input_shape)

        highres_layer = HighRes3DNet(num_classes=5)
        out = highres_layer(x, is_training=True)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 32, 5), out.shape)

    def test_3d_reg_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        x = tf.ones(input_shape)

        highres_layer = HighRes3DNet(
            num_classes=5,
            w_regularizer=regularizers.l2_regularizer(0.5),
            b_regularizer=regularizers.l2_regularizer(0.5))
        out = highres_layer(x, is_training=True)
        out_1 = highres_layer(x, is_training=False)
        # print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 32, 5), out.shape)
            out_1 = sess.run(out_1)
            self.assertAllClose((2, 32, 32, 32, 5), out_1.shape)


if __name__ == "__main__":
    tf.test.main()
