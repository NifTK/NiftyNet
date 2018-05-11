# -*- coding: utf-8 -*-
"""
Creating ``tf.data.Dataset`` instance for image window sampler.
"""
from __future__ import absolute_import, division, print_function

import inspect
import numpy as np
import tensorflow as tf
# pylint: disable=no-name-in-module
from tensorflow.python.data.util import nest

from niftynet.engine.image_window import ImageWindow
from niftynet.engine.image_window import \
    N_SPATIAL, LOCATION_FORMAT, BUFFER_POSITION_NP_TYPE
from niftynet.io.misc_io import squeeze_spatial_temporal_dim
from niftynet.layer.base_layer import Layer


class ImageWindowDataset(Layer):
    """
    This class creates a ``tf.data.Dataset`` instance from
    a sampler's layer_op function or generator.

    If ``from_generator``, ``Dataset.from_generator`` interface will be used,
    ``Dataset.map`` interface will be used otherwise.
    """

    def __init__(self,
                 reader=None,
                 window_sizes=None,
                 batch_size=1,
                 windows_per_image=1,
                 queue_length=10,
                 shuffle=True,
                 epoch=-1,
                 name='image_dataset'):
        Layer.__init__(self, name=name)

        self.dataset = None
        self.iterator = None
        self.reader = reader

        self.batch_size = batch_size
        self.queue_length = queue_length
        self.n_subjects = 1

        self.from_generator = inspect.isgeneratorfunction(self.layer_op)
        self.shuffle = shuffle
        self.epoch = epoch

        self.window = None
        if reader is not None:
            self.window = ImageWindow.from_data_reader_properties(
                reader.input_sources,
                reader.shapes,
                reader.tf_dtypes,
                window_sizes or (-1, -1, -1))
            self.n_subjects = reader.num_subjects
            self.window.n_samples = \
                1 if self.from_generator else windows_per_image
        # OutOfRange error?
        # num_threads?
        # random seeds? (requires num_threads = 1)

    @property
    def shapes(self):
        """
        the sampler output (value of ``layer_op``) is::

            [windows_per_image, x, y, z, 1, channels]

        returns a dictionary of sampler output shapes
        """
        assert self.window, 'Unknown output shapes: self.window not initialised'
        return self.window.shapes

    @property
    def tf_shapes(self):
        """
        returns a dictionary of sampler output tensor shapes
        """
        assert self.window, 'Unknown output shapes: self.window not initialised'
        return self.window.tf_shapes

    @property
    def tf_dtypes(self):
        """
        returns a dictionary of sampler output tensorflow dtypes
        """
        assert self.window, 'Unknown output shapes: self.window not initialised'
        return self.window.tf_dtypes

    def layer_op(self, idx=None):
        """
        Generating each image as a window.
        Overriding this function to create new image sampling strategies.

        This function should either yield a dictionary
        (for single window per image)::

            yield a dictionary

            {
             'image_name': a numpy array,
             'image_name_location': (image_id,
                                     x_start, y_start, z_start,
                                     x_end, y_end, z_end)
            }

        or return a dictionary (for multiple windows per image)::

            return a dictionary:
            {
             'image_name': a numpy array,
             'image_name_location': [n_samples, 7]
            }

        where the 7-element location vector encode the image_id,
        starting and ending coordinates of the image window.

        Following the same notation, the dictionary can be extended
        to multiple modalities; the keys will be::

            {'image_name_1', 'image_name_location_1',
             'image_name_2', 'image_name_location_2', ...}

        :param idx: image_id used to load the image at the i-th row of
            the input
        :return: a image data dictionary
        """

        # dataset: from a window generator
        # assumes self.window.n_samples == 1
        # the generator should yield one window at each iteration
        assert self.window.n_samples == 1, \
            'image_window_dataset.layer_op() requires: ' \
            'windows_per_image should be 1.'
        image_id, image_data, _ = self.reader(idx=idx)
        for mod in list(image_data):
            spatial_shape = image_data[mod].shape[:N_SPATIAL]
            coords = self.dummy_coordinates(image_id, spatial_shape, 1)
            image_data[LOCATION_FORMAT.format(mod)] = coords
            image_data[mod] = image_data[mod][np.newaxis, ...]
        return image_data

    def run_threads(self, session=None, *_unused_args, **_unused_argvs):
        """
        To be called at the beginning of running graph variables,
        to initialise dataset iterator.
        """
        if session is None:
            session = tf.get_default_session()
        if self.dataset is None or self.iterator is None:
            self.init_dataset()
            self.iterator = self.dataset.make_initializable_iterator()
        session.run(self.iterator.initializer)

    def pop_batch_op(self):
        """
        This function is used when connecting a sampler output
        to a network. e.g.::

            data_dict = self.get_sampler()[0].pop_batch_op(device_id)
            net_output = net_model(data_dict['image'], is_training)

        .. caution::

            Note it squeezes the output tensor of 6 dims
            ``[batch, x, y, z, time, modality]``
            by removing all dims along which length is one.

        :return: a dictionary of image window tensors.
        """

        if self.dataset is None or self.iterator is None:
            self.init_dataset()
            self.iterator = self.dataset.make_initializable_iterator()

        window_output = self.iterator.get_next()
        if not self.from_generator:
            window_output = window_output[0]
        # for name in list(self.shapes):
        #     window_output[name].set_shape(
        #         [self.batch_size] + list(self.shapes[name][1:]))
        for name in window_output:
            window_output[name] = squeeze_spatial_temporal_dim(
                window_output[name])
        return window_output

    def init_dataset(self):
        """
        Make a window samples dataset from the reader and layer_op.
        This function sets ``self.dataset``.

        :return:
        """
        if not self.from_generator:
            dataset = self._dataset_from_range()
        else:
            dataset = self._dataset_from_generator()
        self.dataset = self.dataset_preprocessing(dataset)

    def dataset_preprocessing(self, dataset):
        """
        dataset: batch and shuffle

        :param dataset: a `tf.data.Dataset` instance
        :return: a `tf.data.Dataset` instance
        """
        dataset = dataset.repeat(self.epoch)
        buffer_size = int(max(self.queue_length, round(self.batch_size * 3.0)))
        dataset = dataset.prefetch(buffer_size=buffer_size)
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size, seed=None)
        dataset = dataset.batch(batch_size=self.batch_size)
        return dataset

    def _dataset_from_range(self):
        """
        This function maps a dataset of integers to a dataset of images.

        :return: a `tf.data.Dataset`
        """
        # dataset: a list of integers
        dataset = tf.data.Dataset.range(self.n_subjects)

        # dataset: map each integer i to n windows sampled from subject i
        def _tf_wrapper(idx):
            flattened_types = nest.flatten(self.tf_dtypes)
            flattened_shapes = nest.flatten(self.tf_shapes)
            flat_values = tf.py_func(func=self._flatten_layer_op,
                                     inp=[idx],
                                     Tout=flattened_types)
            for ret_t, shape in zip(flat_values, flattened_shapes):
                # the actual returned numpy array shapes are not checked
                ret_t.set_shape(shape)
            return nest.pack_sequence_as(self.tf_dtypes, flat_values)

        dataset = dataset.map(_tf_wrapper, num_parallel_calls=4)

        # dataset: slice the n-element window into n single windows
        def _slice_from_each(*args):
            datasets = [tf.data.Dataset.from_tensor_slices(tensor)
                        for tensor in args]
            return tf.data.Dataset.zip(tuple(datasets))

        dataset = dataset.flat_map(map_func=_slice_from_each)
        return dataset

    def _dataset_from_generator(self):
        """
        Create a `tf.data.Dataset` from a layer_op (as a generator).

        :return: a `tf.data.Dataset`
        """
        win_shapes = {}
        for name in self.tf_shapes:
            win_shapes[name] = self.tf_shapes[name][1:]
        dataset = tf.data.Dataset.from_generator(
            generator=self,
            output_types=self.tf_dtypes,
            output_shapes=win_shapes)
        return dataset

    def _flatten_layer_op(self, idx=None):
        """
        wrapper of the ``layer_op``

        :return: flattened output dictionary as a list
        """
        return nest.flatten(self(idx))

    def close_all(self):
        """
        For compatibility with the queue-based sampler.
        """
        pass

    @classmethod
    def dummy_coordinates(cls, image_id, image_sizes, n_samples):
        """
        This function returns a set of image window coordinates
        which are just from 0 to image_shapes.

        :return: a numpy array of `n_samples` spatial coordinates
        """

        starting_coordinates = [0, 0, 0]
        image_spatial_shape = list(image_sizes[:N_SPATIAL])
        coords = [image_id] + starting_coordinates + image_spatial_shape
        coords = np.tile(np.asarray(coords), [n_samples, 1])
        return coords.astype(BUFFER_POSITION_NP_TYPE)
