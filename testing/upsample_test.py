import tensorflow as tf

from layer.upsample import UpSampleLayer


class UpSampleTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        y = tf.zeros(input_shape)
        x = tf.concat([x, y], 0)

        up_sample_layer = UpSampleLayer('REPLICATE', 3, 3)
        out_up_sample = up_sample_layer(x)
        print up_sample_layer.to_string()
        print out_up_sample.get_shape()

        up_sample_layer = UpSampleLayer('REPLICATE', 2, 2)
        out_up_sample = up_sample_layer(x)
        print up_sample_layer.to_string()
        print out_up_sample.get_shape()

        up_sample_layer = UpSampleLayer('CHANNELWISE_DECONV', 2, 2)
        out_up_sample = up_sample_layer(x)
        print up_sample_layer.to_string()
        print out_up_sample.get_shape()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_up_sample)
            # self.assertAllClose(input_shape, out.shape)
            # self.assertAllClose(np.zeros(input_shape), out)


if __name__ == "__main__":
    tf.test.main()
