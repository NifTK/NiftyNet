import numpy as np
import tensorflow as tf

from layer.input_sampler import ImageSampler


class SamplerTest(tf.test.TestCase):
    def test_shape(self):
        test_sampler = ImageSampler(image_shape=(32, 32, 32),
                                    label_shape=(32, 32, 32),
                                    image_dtype=tf.float32,
                                    label_dtype=tf.int64,
                                    spatial_rank=3,
                                    num_modality=1,
                                    name='sampler')

        print test_sampler.placeholder_names
        print test_sampler.placeholder_dtypes
        print test_sampler.placeholder_shapes
        for data_dict in test_sampler():
            keys = data_dict.keys()[0]
            output = data_dict.values()[0]
            print keys[0], output[0].shape
            print keys[1], output[1].shape
            print keys[2], output[2].shape

        test_sampler = ImageSampler(image_shape=(32, 32, 32),
                                    label_shape=None,
                                    image_dtype=tf.float32,
                                    label_dtype=None,
                                    spatial_rank=3,
                                    num_modality=1,
                                    name='sampler')

        print test_sampler.placeholder_names
        print test_sampler.placeholder_dtypes
        print test_sampler.placeholder_shapes
        for data_dict in test_sampler():
            keys = data_dict.keys()[0]
            output = data_dict.values()[0]
            print keys[0], output[0].shape
            print keys[1], output[1].shape
        #with self.test_session() as sess:
        #    sess.run(tf.global_variables_initializer())
        #    out = sess.run(out_bn)
        #    self.assertAllClose(input_shape, out.shape)
        #    self.assertAllClose(np.zeros(input_shape), out)


if __name__ == "__main__":
    tf.test.main()
