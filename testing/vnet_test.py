import numpy as np
import tensorflow as tf

from layer.vnet import VNet


class VNetTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 32, 32, 32, 1)
        x = tf.ones(input_shape)

        #vnet_instance = VNet(num_classes=160)
        vnet_instance = VNet(num_classes=160)
        out = vnet_instance(x, is_training=True)
        print vnet_instance.num_trainable_params()

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            print out.shape

if __name__ == "__main__":
    tf.test.main()
