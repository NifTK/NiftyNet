import tensorflow as tf

from layer.downsample import DownSampleLayer


class DownSampleTest(tf.test.TestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x = tf.ones(input_shape)
        return x

    def test_3d_max_shape(self):
        x = self.get_3d_input()
        down_sample_layer = DownSampleLayer('MAX', 3, 3)
        out_down_sample_max = down_sample_layer(x)
        print down_sample_layer
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_down_sample_max)
            self.assertAllClose((2, 6, 6, 6, 8), out.shape)

    def test_3d_avg_shape(self):
        x = self.get_3d_input()
        down_sample_layer = DownSampleLayer('AVG', 2, 2)
        out_down_sample_avg = down_sample_layer(x)
        print down_sample_layer
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_down_sample_avg)
            self.assertAllClose((2, 8, 8, 8, 8), out.shape)

    def test_3d_const_shape(self):
        x = self.get_3d_input()
        down_sample_layer = DownSampleLayer('CONSTANT', 3, 3)
        out_down_sample_const = down_sample_layer(x)
        print down_sample_layer
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_down_sample_const)
            self.assertAllClose((2, 6, 6, 6, 8), out.shape)

    def test_2d_max_shape(self):
        x = self.get_2d_input()
        down_sample_layer = DownSampleLayer('MAX', 3, 3)
        out_down_sample_max = down_sample_layer(x)
        print down_sample_layer
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_down_sample_max)
            self.assertAllClose((2, 6, 6, 8), out.shape)

    def test_2d_avg_shape(self):
        x = self.get_2d_input()
        down_sample_layer = DownSampleLayer('AVG', 2, 2)
        out_down_sample_avg = down_sample_layer(x)
        print down_sample_layer
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_down_sample_avg)
            self.assertAllClose((2, 8, 8, 8), out.shape)

    def test_2d_const_shape(self):
        x = self.get_2d_input()
        down_sample_layer = DownSampleLayer('CONSTANT', 3, 3)
        out_down_sample_const = down_sample_layer(x)
        print down_sample_layer
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_down_sample_const)
            self.assertAllClose((2, 6, 6, 8), out.shape)


if __name__ == "__main__":
    tf.test.main()
