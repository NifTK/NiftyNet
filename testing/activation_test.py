import tensorflow as tf

from layer.activation import ActiLayer


class ActivationTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)

        relu_layer = ActiLayer(func='relu')
        out_relu = relu_layer(x)
        print relu_layer

        relu6_layer = ActiLayer(func='relu6')
        out_relu6 = relu6_layer(x)
        print relu6_layer

        elu_layer = ActiLayer(func='elu')
        out_elu = elu_layer(x)
        print elu_layer

        softplus_layer = ActiLayer(func='softplus')
        out_softplus = softplus_layer(x)
        print softplus_layer

        softsign_layer = ActiLayer(func='softsign')
        out_softsign = softsign_layer(x)
        print softsign_layer

        sigmoid_layer = ActiLayer(func='sigmoid')
        out_sigmoid = sigmoid_layer(x)
        print sigmoid_layer

        tanh_layer = ActiLayer(func='tanh')
        out_tanh = tanh_layer(x)
        print tanh_layer

        prelu_layer = ActiLayer(func='prelu')
        out_prelu = prelu_layer(x)
        print prelu_layer

        dropout_layer = ActiLayer(func='dropout')
        out_dropout = dropout_layer(x, keep_prob=0.8)
        print dropout_layer

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_relu)
            out = sess.run(out_relu6)
            out = sess.run(out_elu)
            out = sess.run(out_softplus)
            out = sess.run(out_softsign)
            out = sess.run(out_sigmoid)
            out = sess.run(out_tanh)
            out = sess.run(out_prelu)
            out = sess.run(out_dropout)
            # self.assertAllClose(input_shape, out.shape)
            # self.assertAllClose(np.zeros(input_shape), out)


if __name__ == "__main__":
    tf.test.main()
