from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import ImageWindow
from niftynet.layer.base_layer import Layer


class PairwiseSampler(Layer):
    def __init__(self,
                 reader_0,
                 reader_1,
                 data_param,
                 batch_size,
                 window_per_image):
        Layer.__init__(self, name='pairwise_sampler')
        # reader for the fixed images
        self.reader_0 = reader_0
        # reader for the moving images
        self.reader_1 = reader_1

        # to-do: detect window shape mismatches or defaulting
        # windows to the fixed image reader properties
        self.window = ImageWindow.from_data_reader_properties(
            self.reader_0.input_sources,
            self.reader_0.shapes,
            self.reader_0.tf_dtypes,
            data_param)
        pass

    def run_threads(self, *args, **argvs):
        # do nothing
        pass

    def close_all(self):
        # do nothing
        pass

    def sample_image(self, sampling_reader):
        if sampling_reader == 0:
            image_id_0, data_0, _ = self.reader_0(idx=None, shuffle=True)
            return data_0['fixed_image'].astype(np.float32)
        image_id_1, data_1, _ = self.reader_1(idx=None, shuffle=True)
        return data_1['moving_image'].astype(np.float32)

    def layer_op(self):
        image_0 = tf.py_func(self.sample_image, [tf.constant(0)], tf.float32)
        image_1 = tf.py_func(self.sample_image, [tf.constant(1)], tf.float32)

        affine_augmentation_layer
        return image_1
