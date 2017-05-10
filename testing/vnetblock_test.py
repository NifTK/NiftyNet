import numpy as np
import tensorflow as tf

from layer.vnet import VNetBlock


class VNetBlockTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)

        vnet_block_op = VNetBlock('DOWNSAMPLE', 2, 16, 8)
        out_1, out_2 = vnet_block_op(x, x)
        print vnet_block_op
        print out_1
        print out_2

        vnet_block_op = VNetBlock('UPSAMPLE', 2, 16, 8)
        out_1, out_2 = vnet_block_op(x, x)
        print vnet_block_op
        print out_1
        print out_2

        vnet_block_op = VNetBlock('SAME', 2, 16, 8)
        out_1, out_2 = vnet_block_op(x, x)
        print vnet_block_op
        print out_1
        print out_2

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            out_2 = sess.run(out_2)

if __name__ == "__main__":
    tf.test.main()
