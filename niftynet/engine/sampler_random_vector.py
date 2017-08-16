# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import numpy as np
import scipy.ndimage
import tensorflow as tf

from niftynet.engine.input_buffer import InputBatchQueueRunner
from niftynet.io.image_window import ImageWindow, N_SPATIAL
from niftynet.layer.base_layer import Layer


class RandomVectorSampler(Layer, InputBatchQueueRunner):
    """
    This class generates samples by rescaling the whole image to the desired size
    currently 4D input is supported, Height x Width x Depth x Modality
    """

    def __init__(self, fields=('vector',), vector_size=(100,), batch_size=10):
        # self.reader = reader
        self.n_interpolations = 10
        Layer.__init__(self, name='input_buffer')
        capacity = batch_size * 2

        InputBatchQueueRunner.__init__(self, capacity=capacity, shuffle=False)

        tf.logging.info('reading size of preprocessed images')
        self.fields = fields
        vector_shapes = {fields[0]: vector_size}
        vector_dtypes = {fields[0]: tf.float32}
        self.window = ImageWindow(fields=tuple(vector_shapes),
                                  shapes=vector_shapes,
                                  dtypes=vector_dtypes)
        tf.logging.info('initialised window instance')
        self._create_queue_and_ops(self.window,
                                   enqueue_size=self.n_interpolations,
                                   dequeue_size=self.n_interpolations)
        tf.logging.info("initialised sampler output {} "
                        " [-1 for dynamic size]".format(self.window.shapes))

    def layer_op(self, *args, **kwargs):
        """
        This function generates sampling windows to the input buffer
        image data are from self.reader()
        it first completes window shapes based on image data,
        then finds random coordinates based on the window shapes
        finally resize each image as window and output
        a dictionary (required by input buffer)
        :return: output data dictionary {placeholders: data_array}
        """
        embedding_x = np.random.randn(
            *self.window.shapes[self.window.fields[0]])
        embedding_y = np.random.randn(
            *self.window.shapes[self.window.fields[0]])
        steps = np.linspace(0, 1, self.n_interpolations)
        #enqueue_shape = self.window.shapes[self.fields]
        output_vectors = []
        for (idx, mixture) in enumerate(steps):
            output_vector = embedding_x * mixture + embedding_y * (1-mixture)
            output_vector = output_vector[np.newaxis, ...]
            output_vectors.append(output_vector)
        output_vectors = np.concatenate(output_vectors, axis=0)

        coordinates = np.ones(
                (self.n_interpolations, N_SPATIAL*2+1), dtype=np.int32)

        output_dict = {}
        for name in self.fields:
            coordinates_key = self.window.coordinates_placeholder(name)
            image_data_key = self.window.image_data_placeholder(name)
            output_dict[coordinates_key] = coordinates
            output_dict[image_data_key] = output_vectors
        yield output_dict
