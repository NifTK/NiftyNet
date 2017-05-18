import tensorflow as tf

from layer.deepmedic import DeepMedic


class DeepMedicTest(tf.test.TestCase):
    def test_shape(self):
        input_shape = (2, 57, 57, 57, 1)
        x = tf.ones(input_shape)

        deepmedic_instance = DeepMedic(num_classes=160)
        out = deepmedic_instance(x, is_training=True)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 9, 9, 9, 160), out.shape)

    def test_shape(self):
        input_shape = (2, 57, 57, 1)
        x = tf.ones(input_shape)

        deepmedic_instance = DeepMedic(num_classes=160)
        out = deepmedic_instance(x, is_training=True)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out)
            self.assertAllClose((2, 9, 9, 160), out.shape)

if __name__ == "__main__":
    tf.test.main()
