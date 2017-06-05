from __future__ import absolute_import, print_function

import tensorflow as tf

from layer.upsample import UpSampleLayer


class UpSampleTest(tf.test.TestCase):
    def test_3d_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        y = tf.zeros(input_shape)
        x = tf.concat([x, y], 0)

        up_sample_layer = UpSampleLayer('REPLICATE', 3, 3)
        out_up_sample_rep_1 = up_sample_layer(x)
        print(up_sample_layer.to_string())

        up_sample_layer = UpSampleLayer('REPLICATE', 2, 2)
        out_up_sample_rep_2 = up_sample_layer(x)
        print(up_sample_layer.to_string())

        up_sample_layer = UpSampleLayer('CHANNELWISE_DECONV', 2, 2)
        out_up_sample_deconv = up_sample_layer(x)
        print(up_sample_layer.to_string())

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_up_sample_rep_1)
            self.assertAllClose((4, 48, 48, 48, 8), out.shape)
            out = sess.run(out_up_sample_rep_2)
            self.assertAllClose((4, 32, 32, 32, 8), out.shape)
            out = sess.run(out_up_sample_deconv)
            self.assertAllClose((4, 32, 32, 32, 8), out.shape)
            # self.assertAllClose(np.zeros(input_shape), out)

    def test_2d_shape(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        y = tf.zeros(input_shape)
        x = tf.concat([x, y], 0)

        up_sample_layer = UpSampleLayer('REPLICATE', 3, 3)
        out_up_sample_rep_1 = up_sample_layer(x)
        print(up_sample_layer.to_string())

        up_sample_layer = UpSampleLayer('REPLICATE', 2, 2)
        out_up_sample_rep_2 = up_sample_layer(x)
        print(up_sample_layer.to_string())

        up_sample_layer = UpSampleLayer('CHANNELWISE_DECONV', 2, 2)
        out_up_sample_deconv = up_sample_layer(x)
        print(up_sample_layer.to_string())

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_up_sample_rep_1)
            self.assertAllClose((4, 48, 48, 8), out.shape)
            out = sess.run(out_up_sample_rep_2)
            self.assertAllClose((4, 32, 32, 8), out.shape)
            out = sess.run(out_up_sample_deconv)
            self.assertAllClose((4, 32, 32, 8), out.shape)


if __name__ == "__main__":
    tf.test.main()
