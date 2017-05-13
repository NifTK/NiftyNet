import tensorflow as tf

from layer.unet import UNet3D


class UNet3DTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        x = tf.ones(input_shape)

        unet_instance = UNet3D(num_classes=160)
        out = unet_instance(x, is_training=True)
        print unet_instance.num_trainable_params()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            print out.shape


if __name__ == "__main__":
    tf.test.main()
