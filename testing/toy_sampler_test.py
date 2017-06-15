from __future__ import absolute_import, print_function

import tensorflow as tf

from engine.toy_sampler import ToySampler
from utilities.input_placeholders import ImagePatch


class ToySamplerTest(tf.test.TestCase):
    def run_test_sampler(self, patch_holder=None):
        test_sampler = ToySampler(patch_holder, name='sampler')
        print(test_sampler.placeholder_names)
        print(test_sampler.placeholder_dtypes)
        print(test_sampler.placeholder_shapes)
        for d in test_sampler():
            assert isinstance(d, ImagePatch)
            data_dict = d.as_dict(test_sampler.placeholders)
            keys = list(data_dict)[0]
            output = data_dict[keys][0]
            for (idx, key) in enumerate(keys):
                print(key, output[idx].shape)

    def test_full_shape_3d(self):
        patch_holder = ImagePatch(spatial_rank=3,
                                  image_size=32,
                                  label_size=32,
                                  weight_map_size=32,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_image_modality=1,
                                  num_label_modality=1,
                                  num_weight_map=1)
        self.run_test_sampler(patch_holder)

    def test_image_label_info_shape_3d(self):
        patch_holder = ImagePatch(spatial_rank=3,
                                  image_size=32,
                                  label_size=32,
                                  weight_map_size=None,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_image_modality=1,
                                  num_label_modality=1,
                                  num_weight_map=1)
        self.run_test_sampler(patch_holder)

    def test_image_weight_map_info_shape_3d(self):
        patch_holder = ImagePatch(spatial_rank=3,
                                  image_size=32,
                                  label_size=None,
                                  weight_map_size=32,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_image_modality=1,
                                  num_label_modality=1,
                                  num_weight_map=1)
        self.run_test_sampler(patch_holder)

    def test_image_info_shape_3d(self):
        patch_holder = ImagePatch(spatial_rank=3,
                                  image_size=32,
                                  label_size=None,
                                  weight_map_size=None,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_image_modality=1,
                                  num_label_modality=1,
                                  num_weight_map=1)
        self.run_test_sampler(patch_holder)

    def test_full_shape_2d(self):
        patch_holder = ImagePatch(spatial_rank=2,
                                  image_size=32,
                                  label_size=32,
                                  weight_map_size=32,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_image_modality=1,
                                  num_label_modality=1,
                                  num_weight_map=1)
        self.run_test_sampler(patch_holder)

    def test_image_label_info_shape_2d(self):
        patch_holder = ImagePatch(spatial_rank=2,
                                  image_size=32,
                                  label_size=32,
                                  weight_map_size=None,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_image_modality=1,
                                  num_label_modality=1,
                                  num_weight_map=1)
        self.run_test_sampler(patch_holder)

    def test_image_weight_map_info_shape_2d(self):
        patch_holder = ImagePatch(spatial_rank=2,
                                  image_size=32,
                                  label_size=None,
                                  weight_map_size=32,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_image_modality=1,
                                  num_label_modality=1,
                                  num_weight_map=1)
        self.run_test_sampler(patch_holder)

    def test_image_info_shape_2d(self):
        patch_holder = ImagePatch(spatial_rank=2,
                                  image_size=32,
                                  label_size=None,
                                  weight_map_size=None,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_image_modality=1,
                                  num_label_modality=1,
                                  num_weight_map=1)
        self.run_test_sampler(patch_holder)


if __name__ == "__main__":
    tf.test.main()
