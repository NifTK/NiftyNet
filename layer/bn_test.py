import numpy as np
import tensorflow as tf

from bn import BNLayer


class BNTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        bn_layer = BNLayer()
        out_bn = bn_layer(x, is_training=True)
        print bn_layer.to_string()
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_bn)
            self.assertAllClose(input_shape, out.shape)
            self.assertAllClose(np.zeros(input_shape), out)

if __name__ == "__main__":
    tf.test.main()
