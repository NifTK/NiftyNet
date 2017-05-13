import tensorflow as tf

from layer.deconvolution import DeconvLayer
from layer.deconvolution import DeconvolutionalLayer


class DeconvTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x_3d = tf.ones(input_shape)

        input_shape = (2, 16, 16, 8)
        x_2d = tf.ones(input_shape)

        deconv_3d = DeconvLayer(10, 3, 2, with_bias=True, padding='VALID')
        conv_3d_out = deconv_3d(x_3d)
        print deconv_3d
        print conv_3d_out.get_shape()

        deconv_bn = DeconvolutionalLayer(10, 3, 1)
        conv_bn_out = deconv_bn(x_3d, is_training=True)
        print deconv_bn
        print conv_bn_out.get_shape()

        deconv_bn_relu = DeconvolutionalLayer(10, 3, 1, acti_fun='relu')
        conv_bn_relu_out = deconv_bn_relu(x_3d, is_training=True, keep_prob=0.8)
        print deconv_bn_relu
        print conv_bn_relu_out.get_shape()

        deconv_2d = DeconvLayer(10, 3, 1)
        conv_2d_out = deconv_2d(x_2d)
        print deconv_2d
        print conv_2d_out.get_shape()

        deconv_reg = DeconvLayer(10, 3, 1, w_regularizer=tf.nn.l2_loss)
        conv_reg_out = deconv_reg(x_2d)
        # print tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        print conv_reg_out.get_shape()

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
