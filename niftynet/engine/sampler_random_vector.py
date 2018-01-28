# -*- coding: utf-8 -*-
"""
Generating sample arrays from random distributions.
"""
from __future__ import absolute_import, print_function, division

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import ImageWindow, N_SPATIAL
from niftynet.engine.image_window_buffer import InputBatchQueueRunner
from niftynet.layer.base_layer import Layer


class RandomVectorSampler(Layer, InputBatchQueueRunner):
    """
    This class generates two samples from the standard normal
    distribution.  These two samples are mixed with n
    mixing coefficients. The coefficients are generated
    by ``np.linspace(0, 1, n_interpolations)``
    """

    def __init__(self,
                 names=('vector',),
                 vector_size=(100,),
                 batch_size=10,
                 n_interpolations=10,
                 mean=0.0,
                 stddev=1.0,
                 repeat=1,
                 queue_length=10):
        self.n_interpolations = max(n_interpolations, 1)
        self.mean = mean
        self.stddev = stddev
        self.repeat = repeat
        Layer.__init__(self, name='input_buffer')
        InputBatchQueueRunner.__init__(
            self,
            capacity=queue_length,
            shuffle=False)
        tf.logging.info('reading size of preprocessed images')
        self.names = names
        vector_shapes = {names[0]: vector_size}
        vector_dtypes = {names[0]: tf.float32}
        self.window = ImageWindow(names=tuple(vector_shapes),
                                  shapes=vector_shapes,
                                  dtypes=vector_dtypes)
        tf.logging.info('initialised window instance')
        self._create_queue_and_ops(self.window,
                                   enqueue_size=self.n_interpolations,
                                   dequeue_size=batch_size)
        tf.logging.info("initialised sampler output %s ", self.window.shapes)

    def layer_op(self, *args, **kwargs):
        """
        This function first draws two samples, and interpolates them
        with self.n_interpolations mixing coefficients.

        Location coordinates are set to ``np.ones`` for all the vectors.
        """
        total_iter = self.repeat if self.repeat is not None else 1
        while total_iter > 0:
            total_iter = total_iter - 1 if self.repeat is not None else 1
            embedding_x = np.random.normal(
                self.mean,
                self.stddev,
                *self.window.shapes[self.window.names[0]])
            embedding_y = np.random.normal(
                self.mean,
                self.stddev,
                *self.window.shapes[self.window.names[0]])
            steps = np.linspace(0, 1, self.n_interpolations)
            output_vectors = []
            for (_, mixture) in enumerate(steps):
                output_vector = \
                    embedding_x * mixture + embedding_y * (1 - mixture)
                output_vector = output_vector[np.newaxis, ...]
                output_vectors.append(output_vector)
            output_vectors = np.concatenate(output_vectors, axis=0)

            coordinates = np.ones(
                (self.n_interpolations, N_SPATIAL * 2 + 1), dtype=np.int32)

            output_dict = {}
            for name in self.window.names:
                coordinates_key = self.window.coordinates_placeholder(name)
                image_data_key = self.window.image_data_placeholder(name)
                output_dict[coordinates_key] = coordinates
                output_dict[image_data_key] = output_vectors
            yield output_dict
