import numpy as np
import tensorflow as tf

from layer.scalenet import ScaleBlock


class ScaleBlockTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 32, 32, 32, 4)
        x = tf.ones(input_shape)
        x = tf.unstack(x, axis=-1)
        for (idx, fea) in enumerate(x):
            x[idx] = tf.expand_dims(fea, axis=-1)
        x = tf.stack(x, axis=-1)

        scalenet_layer = ScaleBlock('AVERAGE', n_layers=1)
        out_1 = scalenet_layer(x, is_training=True)
        print scalenet_layer

        scalenet_layer = ScaleBlock('MAX', n_layers=2)
        out_2 = scalenet_layer(x, is_training=True)
        print scalenet_layer

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out_1 = sess.run(out_1)
            out_2 = sess.run(out_2)
            print out_1.shape
            print out_2.shape

if __name__ == "__main__":
    tf.test.main()
