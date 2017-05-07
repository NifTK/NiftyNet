import numpy as np
import tensorflow as tf

from convolution import ConvLayer


class ConvTest(tf.test.TestCase):
    def test_shape(self):
        print 'convolution_test'
        #conv_3d = ConvLayer(tf.nn.conv3d, 10)
        #conv_3d = ConvLayer(tf.nn.conv3d, 10, b_initializer=tf.ones_initializer)
        conv_3d = ConvLayer(tf.nn.conv3d, 10, [3, 3, 3], [1, 1, 1, 1, 1])
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        conv_3d_out = conv_3d(x)
        print conv_3d

        conv_2d = ConvLayer(tf.nn.conv2d, 10, [3, 3], [1, 1, 1, 1])
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        conv_2d_out = conv_2d(x)
        print conv_2d

        conv_reg = ConvLayer(tf.nn.conv2d, 10, [3, 3], [1, 1, 1, 1],
                w_regularizer=tf.nn.l2_loss)
        reg_out = conv_reg(x)
        print tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_2d_out)
            out_3d = sess.run(conv_3d_out)
        #    self.assertAllClose(input_shape, out.shape)
        #    self.assertAllClose(np.zeros(input_shape), out)

if __name__ == "__main__":
    tf.test.main()
