import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from layer.convolution import ConvLayer
from layer.convolution import ConvolutionalLayer


class ConvTest(tf.test.TestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x_3d = tf.ones(input_shape)
        return x_3d

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x_2d = tf.ones(input_shape)
        return x_2d

    # 3d tests
    def test_3d_conv_default_shape(self):
        x_3d = self.get_3d_input()
        conv_3d = ConvLayer(10, 3, 1)
        conv_3d_out = conv_3d(x_3d)
        print(conv_3d)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_3d = sess.run(conv_3d_out)
            self.assertAllClose((2, 16, 16, 16, 10), out_3d.shape)

    def test_3d_conv_bias_shape(self):
        x_3d = self.get_3d_input()
        conv_3d = ConvLayer(10, 3, 1, with_bias=True)
        conv_3d_out = conv_3d(x_3d)
        print(conv_3d)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_3d = sess.run(conv_3d_out)
            self.assertAllClose((2, 16, 16, 16, 10), out_3d.shape)

    def test_conv_3d_bias_reg_shape(self):
        x_3d = self.get_3d_input()
        conv_reg = ConvLayer(10, 3, 2,
                             w_regularizer=regularizers.l2_regularizer(0.5),
                             with_bias=True,
                             b_regularizer=regularizers.l2_regularizer(0.5))
        conv_reg_out = conv_reg(x_3d)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_3d = sess.run(conv_reg_out)
            self.assertAllClose((2, 8, 8, 8, 10), out_3d.shape)

    def test_3d_convlayer_default_shape(self):
        x_3d = self.get_3d_input()
        conv_3d = ConvolutionalLayer(10, 3, 1)
        conv_3d_out = conv_3d(x_3d, is_training=True)
        print(conv_3d)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_3d = sess.run(conv_3d_out)
            self.assertAllClose((2, 16, 16, 16, 10), out_3d.shape)

    def test_3d_convlayer_bias_shape(self):
        x_3d = self.get_3d_input()
        conv_3d = ConvolutionalLayer(10, 3, 1, with_bias=True, with_bn=False)
        conv_3d_out = conv_3d(x_3d)
        print(conv_3d)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_3d = sess.run(conv_3d_out)
            self.assertAllClose((2, 16, 16, 16, 10), out_3d.shape)

    def test_convlayer_3d_bias_reg_shape(self):
        x_3d = self.get_3d_input()
        conv_reg = ConvolutionalLayer(
            10, 3, 1,
            w_regularizer=regularizers.l2_regularizer(0.5),
            with_bias=True,
            b_regularizer=regularizers.l2_regularizer(0.5),
            with_bn=False)
        conv_reg_out = conv_reg(x_3d)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_3d = sess.run(conv_reg_out)
            self.assertAllClose((2, 16, 16, 16, 10), out_3d.shape)

    def test_convlayer_3d_bn_reg_shape(self):
        x_3d = self.get_3d_input()
        conv_reg = ConvolutionalLayer(
            10, 3, 1,
            w_regularizer=regularizers.l2_regularizer(0.5),
            with_bias=False,
            with_bn=True)
        conv_reg_out = conv_reg(x_3d, is_training=True)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_3d = sess.run(conv_reg_out)
            self.assertAllClose((2, 16, 16, 16, 10), out_3d.shape)

    def test_convlayer_3d_bn_reg_prelu_shape(self):
        x_3d = self.get_3d_input()
        conv_reg = ConvolutionalLayer(
            10, 3, 2,
            w_regularizer=regularizers.l2_regularizer(0.5),
            with_bias=False,
            with_bn=True,
            acti_func='prelu')
        conv_reg_out = conv_reg(x_3d, is_training=True)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_3d = sess.run(conv_reg_out)
            self.assertAllClose((2, 8, 8, 8, 10), out_3d.shape)

    def test_convlayer_3d_relu_shape(self):
        x_3d = self.get_3d_input()
        conv_reg = ConvolutionalLayer(
            10, 3, 1,
            acti_func='relu')
        conv_reg_out = conv_reg(x_3d, is_training=True)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_3d = sess.run(conv_reg_out)
            self.assertAllClose((2, 16, 16, 16, 10), out_3d.shape)

    def test_convlayer_3d_bn_reg_dropout_shape(self):
        x_3d = self.get_3d_input()
        conv_reg = ConvolutionalLayer(
            10, 3, 2,
            w_regularizer=regularizers.l2_regularizer(0.5),
            with_bias=False,
            with_bn=True,
            acti_func='prelu')
        conv_reg_out = conv_reg(x_3d, is_training=True, keep_prob=0.4)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_3d = sess.run(conv_reg_out)
            self.assertAllClose((2, 8, 8, 8, 10), out_3d.shape)

    def test_convlayer_3d_bn_reg_dropout_valid_shape(self):
        x_3d = self.get_3d_input()
        conv_reg = ConvolutionalLayer(
            10, 3, 1,
            padding='VALID',
            w_regularizer=regularizers.l2_regularizer(0.5),
            with_bias=False,
            with_bn=True,
            acti_func='prelu')
        conv_reg_out = conv_reg(x_3d, is_training=True, keep_prob=0.4)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_3d = sess.run(conv_reg_out)
            self.assertAllClose((2, 14, 14, 14, 10), out_3d.shape)

    # 2d tests
    def test_2d_conv_default_shape(self):
        x_2d = self.get_2d_input()
        conv_2d = ConvLayer(10, 3, 1)
        conv_2d_out = conv_2d(x_2d)
        print(conv_2d)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_2d_out)
            self.assertAllClose((2, 16, 16, 10), out_2d.shape)

    def test_2d_conv_bias_shape(self):
        x_2d = self.get_2d_input()
        conv_2d = ConvLayer(10, 3, 1, with_bias=True)
        conv_2d_out = conv_2d(x_2d)
        print(conv_2d)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_2d_out)
            self.assertAllClose((2, 16, 16, 10), out_2d.shape)

    def test_conv_2d_bias_reg_shape(self):
        x_2d = self.get_2d_input()
        conv_reg = ConvLayer(10, 3, 1,
                             w_regularizer=regularizers.l2_regularizer(0.5),
                             with_bias=True,
                             b_regularizer=regularizers.l2_regularizer(0.5))
        conv_reg_out = conv_reg(x_2d)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_reg_out)
            self.assertAllClose((2, 16, 16, 10), out_2d.shape)

    def test_2d_convlayer_default_shape(self):
        x_2d = self.get_2d_input()
        conv_2d = ConvolutionalLayer(10, 3, 1)
        conv_2d_out = conv_2d(x_2d, is_training=True)
        print(conv_2d)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_2d_out)
            self.assertAllClose((2, 16, 16, 10), out_2d.shape)

    def test_2d_convlayer_bias_shape(self):
        x_2d = self.get_2d_input()
        conv_2d = ConvolutionalLayer(10, 3, 1, with_bias=True, with_bn=False)
        conv_2d_out = conv_2d(x_2d)
        print(conv_2d)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_2d_out)
            self.assertAllClose((2, 16, 16, 10), out_2d.shape)

    def test_convlayer_2d_bias_reg_shape(self):
        x_2d = self.get_2d_input()
        conv_reg = ConvolutionalLayer(
            10, 3, 1,
            w_regularizer=regularizers.l2_regularizer(0.5),
            with_bias=True,
            b_regularizer=regularizers.l2_regularizer(0.5),
            with_bn=False)
        conv_reg_out = conv_reg(x_2d)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_reg_out)
            self.assertAllClose((2, 16, 16, 10), out_2d.shape)

    def test_convlayer_2d_bn_reg_shape(self):
        x_2d = self.get_2d_input()
        conv_reg = ConvolutionalLayer(
            10, 3, 1,
            w_regularizer=regularizers.l2_regularizer(0.5),
            with_bias=False,
            with_bn=True)
        conv_reg_out = conv_reg(x_2d, is_training=True)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_reg_out)
            self.assertAllClose((2, 16, 16, 10), out_2d.shape)

    def test_convlayer_2d_bn_reg_prelu_2_shape(self):
        x_2d = self.get_2d_input()
        conv_reg = ConvolutionalLayer(
            10, 3, 2,
            w_regularizer=regularizers.l2_regularizer(0.5),
            with_bias=False,
            with_bn=True,
            acti_func='prelu')
        conv_reg_out = conv_reg(x_2d, is_training=True)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_reg_out)
            self.assertAllClose((2, 8, 8, 10), out_2d.shape)

    def test_convlayer_2d_relu_shape(self):
        x_2d = self.get_2d_input()
        conv_reg = ConvolutionalLayer(
            10, 3, 2,
            acti_func='relu')
        conv_reg_out = conv_reg(x_2d, is_training=True)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_reg_out)
            self.assertAllClose((2, 8, 8, 10), out_2d.shape)

    def test_convlayer_2d_bn_reg_prelu_shape(self):
        x_2d = self.get_2d_input()
        conv_reg = ConvolutionalLayer(
            10, 3, 1,
            w_regularizer=regularizers.l2_regularizer(0.5),
            with_bias=False,
            with_bn=True,
            acti_func='prelu')
        conv_reg_out = conv_reg(x_2d, is_training=True, keep_prob=0.4)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_reg_out)
            self.assertAllClose((2, 16, 16, 10), out_2d.shape)

    def test_convlayer_2d_bn_reg_valid_shape(self):
        x_2d = self.get_2d_input()
        conv_reg = ConvolutionalLayer(
            10, 3, 1,
            padding='VALID',
            w_regularizer=regularizers.l2_regularizer(0.5),
            with_bias=False,
            with_bn=True,
            acti_func='prelu')
        conv_reg_out = conv_reg(x_2d, is_training=True, keep_prob=0.4)
        print(conv_reg)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_2d = sess.run(conv_reg_out)
            self.assertAllClose((2, 14, 14, 10), out_2d.shape)


if __name__ == "__main__":
    tf.test.main()
