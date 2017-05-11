import numpy as np
import tensorflow as tf

from layer.unet import UNetBlock


class UNetBlockTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)

        unet_block_op = UNetBlock(
                'DOWNSAMPLE', (32, 64), (3, 3), with_downsample_branch=True)
        out_1, out_2 = unet_block_op(x, is_training=True)
        print unet_block_op
        print out_1
        print out_2

        unet_block_op = UNetBlock(
                'UPSAMPLE', (32, 64), (3, 3), with_downsample_branch=False)
        out_3, _ = unet_block_op(x, is_training=True)
        print unet_block_op
        print out_3

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            out_2 = sess.run(out_2)
            out_3 = sess.run(out_3)

if __name__ == "__main__":
    tf.test.main()
