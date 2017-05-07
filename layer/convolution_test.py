import numpy as np
import tensorflow as tf

from convolution import ConvLayer
from convolution import ConvBNLayer


class ConvTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x_3d = tf.ones(input_shape)

        input_shape = (2, 16, 16, 8)
        x_2d = tf.ones(input_shape)

        conv_3d = ConvLayer('3D', 10, 3, 1, with_bias=True)
        conv_3d_out = conv_3d(x_3d)
        print conv_3d

        conv_bn = ConvBNLayer('3D', 10, 3, 1)
        conv_bn_out = conv_bn(x_3d, is_training=True)
        print conv_bn

        conv_bn_relu = ConvBNLayer('3D', 10, 3, 1, acti_fun='prelu')
        conv_bn_relu_out = conv_bn_relu(x_3d, is_training=True, keep_prob=0.8)
        print conv_bn_relu

        conv_2d = ConvLayer('2D', 10, 3, 1)
        conv_2d_out = conv_2d(x_2d)
        print conv_2d

        conv_reg = ConvLayer('2D', 10, 3, 1, w_regularizer=tf.nn.l2_loss)
        conv_reg_out = conv_reg(x_2d)
        print tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)


        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_2d_out)
            out_3d = sess.run(conv_3d_out)
            out_bn = sess.run(conv_bn_out)
            out_bn_relu = sess.run(conv_bn_relu_out)
            out_reg = sess.run(conv_reg_out)
        #    self.assertAllClose(input_shape, out.shape)
        #    self.assertAllClose(np.zeros(input_shape), out)

if __name__ == "__main__":
    tf.test.main()
