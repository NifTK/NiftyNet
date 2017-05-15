import numpy as np
import tensorflow as tf

from layer.input_sampler import ImageSampler
from layer.input_placeholders import ImagePatch


class SamplerTest(tf.test.TestCase):

    def run_test_sampler(self, patch_holder=None):
        if patch_holder is None:
            return
        test_sampler = ImageSampler(patch_holder, name='sampler')
        print test_sampler.placeholder_names
        print test_sampler.placeholder_dtypes
        print test_sampler.placeholder_shapes
        for data_dict in test_sampler():
            keys = data_dict.keys()[0]
            output = data_dict.values()[0]
            for (idx, key) in enumerate(keys):
                print key, output[idx].shape


    def test_full_shape_3d(self):
        patch_holder = ImagePatch(image_shape=(32, 32, 32),
                                  label_shape=(32, 32, 32),
                                  weight_map_shape=(32, 32, 32),
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_modality=1,
                                  num_map=1,
                                  name='patch_placeholders')
        self.run_test_sampler(patch_holder)


    def test_image_label_info_shape_3d(self):
        patch_holder = ImagePatch(image_shape=(32, 32, 32),
                                  label_shape=(32, 32, 32),
                                  weight_map_shape=None,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_modality=1,
                                  num_map=1,
                                  name='patch_placeholders_1')
        self.run_test_sampler(patch_holder)

    def test_image_weight_map_info_shape_3d(self):
        patch_holder = ImagePatch(image_shape=(32, 32, 32),
                                  label_shape=None,
                                  weight_map_shape=(32, 32, 32),
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_modality=1,
                                  num_map=1,
                                  name='patch_placeholders_1')
        self.run_test_sampler(patch_holder)

    def test_image_info_shape_3d(self):
        patch_holder = ImagePatch(image_shape=(32, 32, 32),
                                  label_shape=None,
                                  weight_map_shape=None,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_modality=1,
                                  num_map=1,
                                  name='patch_placeholders_1')
        self.run_test_sampler(patch_holder)

    def test_full_shape_2d(self):
        patch_holder = ImagePatch(image_shape=(32, 32),
                                  label_shape=(32, 32),
                                  weight_map_shape=(32, 32),
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_modality=1,
                                  num_map=1,
                                  name='patch_placeholders')
        self.run_test_sampler(patch_holder)


    def test_image_label_info_shape_2d(self):
        patch_holder = ImagePatch(image_shape=(32, 32),
                                  label_shape=(32, 32),
                                  weight_map_shape=None,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_modality=1,
                                  num_map=1,
                                  name='patch_placeholders_1')
        self.run_test_sampler(patch_holder)

    def test_image_weight_map_info_shape_2d(self):
        patch_holder = ImagePatch(image_shape=(32, 32),
                                  label_shape=None,
                                  weight_map_shape=(32, 32),
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_modality=1,
                                  num_map=1,
                                  name='patch_placeholders_1')
        self.run_test_sampler(patch_holder)

    def test_image_info_shape_2d(self):
        patch_holder = ImagePatch(image_shape=(32, 32),
                                  label_shape=None,
                                  weight_map_shape=None,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_modality=1,
                                  num_map=1,
                                  name='patch_placeholders_1')
        self.run_test_sampler(patch_holder)


if __name__ == "__main__":
    tf.test.main()
