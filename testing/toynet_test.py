import numpy as np
import tensorflow as tf

from layer.toynet import ToyNet


class ToyNetTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        x = tf.ones(input_shape)

        toynet_instance = ToyNet(num_classes=160)
        out = toynet_instance(x, is_training=True)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            print out.shape

if __name__ == "__main__":
    tf.test.main()
