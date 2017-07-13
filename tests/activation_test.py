from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.layer.activation import ActiLayer


class ActivationTest(tf.test.TestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    # 3d test
    def test_3d_relu_shape(self):
        x = self.get_3d_input()
        relu_layer = ActiLayer(func='relu')
        out_relu = relu_layer(x)
        print(relu_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_relu)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_relu6_shape(self):
        x = self.get_3d_input()
        relu6_layer = ActiLayer(func='relu6')
        out_relu6 = relu6_layer(x)
        print(relu6_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_relu6)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_elu_shape(self):
        x = self.get_3d_input()
        elu_layer = ActiLayer(func='elu')
        out_elu = elu_layer(x)
        print(elu_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_elu)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_softplus_shape(self):
        x = self.get_3d_input()
        softplus_layer = ActiLayer(func='softplus')
        out_softplus = softplus_layer(x)
        print(softplus_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_softplus)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_softsign_shape(self):
        x = self.get_3d_input()
        softsign_layer = ActiLayer(func='softsign')
        out_softsign = softsign_layer(x)
        print(softsign_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_softsign)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_sigmoid_shape(self):
        x = self.get_3d_input()
        sigmoid_layer = ActiLayer(func='sigmoid')
        out_sigmoid = sigmoid_layer(x)
        print(sigmoid_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_sigmoid)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_tanh_shape(self):
        x = self.get_3d_input()
        tanh_layer = ActiLayer(func='tanh')
        out_tanh = tanh_layer(x)
        print(tanh_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_tanh)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_prelu_shape(self):
        x = self.get_3d_input()
        prelu_layer = ActiLayer(func='prelu')
        out_prelu = prelu_layer(x)
        print(prelu_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_prelu)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_prelu_reg_shape(self):
        x = self.get_3d_input()
        prelu_layer = ActiLayer(func='prelu',
                                regularizer=regularizers.l2_regularizer(0.5),
                                name='regularized')
        out_prelu = prelu_layer(x)
        print(prelu_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_prelu)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_identity_shape(self):
        x = self.get_3d_input()
        identity_layer = ActiLayer(func='identity')
        out_prelu = identity_layer(x)
        print(identity_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_prelu)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

    def test_3d_dropout_shape(self):
        x = self.get_3d_input()
        dropout_layer = ActiLayer(func='dropout')
        out_dropout = dropout_layer(x, keep_prob=0.8)
        print(dropout_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_dropout)
            self.assertAllClose((2, 16, 16, 16, 8), out.shape)

            # self.assertAllClose(input_shape, out.shape)
            # self.assertAllClose(np.zeros(input_shape), out)

    # 2d test
    def test_2d_relu_shape(self):
        x = self.get_2d_input()
        relu_layer = ActiLayer(func='relu')
        out_relu = relu_layer(x)
        print(relu_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_relu)
            self.assertAllClose((2, 16, 16, 8), out.shape)

    def test_2d_relu6_shape(self):
        x = self.get_2d_input()
        relu6_layer = ActiLayer(func='relu6')
        out_relu6 = relu6_layer(x)
        print(relu6_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_relu6)
            self.assertAllClose((2, 16, 16, 8), out.shape)

    def test_2d_elu_shape(self):
        x = self.get_2d_input()
        elu_layer = ActiLayer(func='elu')
        out_elu = elu_layer(x)
        print(elu_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_elu)
            self.assertAllClose((2, 16, 16, 8), out.shape)

    def test_2d_softplus_shape(self):
        x = self.get_2d_input()
        softplus_layer = ActiLayer(func='softplus')
        out_softplus = softplus_layer(x)
        print(softplus_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_softplus)
            self.assertAllClose((2, 16, 16, 8), out.shape)

    def test_2d_softsign_shape(self):
        x = self.get_2d_input()
        softsign_layer = ActiLayer(func='softsign')
        out_softsign = softsign_layer(x)
        print(softsign_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_softsign)
            self.assertAllClose((2, 16, 16, 8), out.shape)

    def test_2d_sigmoid_shape(self):
        x = self.get_2d_input()
        sigmoid_layer = ActiLayer(func='sigmoid')
        out_sigmoid = sigmoid_layer(x)
        print(sigmoid_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_sigmoid)
            self.assertAllClose((2, 16, 16, 8), out.shape)

    def test_2d_tanh_shape(self):
        x = self.get_2d_input()
        tanh_layer = ActiLayer(func='tanh')
        out_tanh = tanh_layer(x)
        print(tanh_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_tanh)
            self.assertAllClose((2, 16, 16, 8), out.shape)

    def test_2d_prelu_shape(self):
        x = self.get_2d_input()
        prelu_layer = ActiLayer(func='prelu')
        out_prelu = prelu_layer(x)
        print(prelu_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_prelu)
            self.assertAllClose((2, 16, 16, 8), out.shape)

    def test_2d_prelu_reg_shape(self):
        x = self.get_2d_input()
        prelu_layer = ActiLayer(func='prelu',
                                regularizer=regularizers.l2_regularizer(0.5),
                                name='regularized')
        out_prelu = prelu_layer(x)
        print(prelu_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_prelu)
            self.assertAllClose((2, 16, 16, 8), out.shape)

    def test_2d_identity_shape(self):
        x = self.get_2d_input()
        identity_layer = ActiLayer(func='identity')
        out_dropout = identity_layer(x, keep_prob=0.8)
        print(identity_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_dropout)
            self.assertAllClose((2, 16, 16, 8), out.shape)

    def test_2d_dropout_shape(self):
        x = self.get_2d_input()
        dropout_layer = ActiLayer(func='dropout')
        out_dropout = dropout_layer(x, keep_prob=0.8)
        print(dropout_layer)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_dropout)
            self.assertAllClose((2, 16, 16, 8), out.shape)


if __name__ == "__main__":
    tf.test.main()
