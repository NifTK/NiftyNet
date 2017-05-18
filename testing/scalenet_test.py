import tensorflow as tf

from layer.scalenet import ScaleNet


class ScaleNetTest(tf.test.TestCase):
    def test_3d_shape(self):
        input_shape = (2, 32, 32, 32, 4)
        x = tf.ones(input_shape)

        scalenet_layer = ScaleNet(num_classes=5)
        out = scalenet_layer(x, is_training=True)
        print scalenet_layer.num_trainable_params()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 32, 5), out.shape)

    def test_2d_shape(self):
        input_shape = (2, 32, 32, 4)
        x = tf.ones(input_shape)

        scalenet_layer = ScaleNet(num_classes=5)
        out = scalenet_layer(x, is_training=True)
        print scalenet_layer.num_trainable_params()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 32, 32, 5), out.shape)

if __name__ == "__main__":
    tf.test.main()
