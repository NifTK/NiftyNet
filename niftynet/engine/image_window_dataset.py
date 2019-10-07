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
from tensorflow.python.keras.utils import GeneratorEnqueuer

from niftynet.engine.image_window import ImageWindow, N_SPATIAL, \
    LOCATION_FORMAT, BUFFER_POSITION_NP_TYPE
from niftynet.io.misc_io import squeeze_spatial_temporal_dim
from niftynet.layer.base_layer import Layer
from niftynet.utilities.util_common import look_up_operations

# when total number of window samples are not divisible by batch_size
# the class supports different modes for the final batch
#   'drop': drop the remainder batch
#   'pad': padding the final smaller batch with -1s
#   'dynamic': output the remainder directly (in this case the batch_size
#              is undetermined at 'compile time')
SMALLER_FINAL_BATCH_MODE = {'drop', 'pad', 'dynamic'}


# pylint: disable=too-many-instance-attributes
class ImageWindowDataset(Layer):
    """
    This class creates a ``tf.data.Dataset`` instance from
    a sampler's layer_op function or generator.

    If ``from_generator``, ``Dataset.from_generator`` interface will be used,
    ``Dataset.map`` interface will be used otherwise::

        if the windows are from a image reader,
        the total number of windows produced
        will be: `epoch x n_subjects x windows_per_image`

        if the windows are from a generator,
        the total number of windows produced
        will be: "iterations from the generator" x num_threads

    """

    # pylint: disable=too-many-arguments
    def __init__(self,
                 reader=None,
                 window_sizes=None,
                 batch_size=1,
                 windows_per_image=1,
                 queue_length=10,
                 shuffle=True,
                 epoch=-1,
                 smaller_final_batch_mode='pad',
                 seed=None,
                 name='image_dataset'):
        Layer.__init__(self, name=name)

        self._num_threads = 1
        self._enqueuer = None
        self._seed = seed

        self.dataset = None
        self.iterator = None
        self.reader = reader

        self.batch_size = batch_size
        self.queue_length = int(max(queue_length, round(batch_size * 5.0)))
        if self.queue_length > queue_length:
            tf.logging.warning(
                'sampler queue_length should be larger than batch_size, '
                'defaulting to batch_size * 5.0 (%s).', self.queue_length)

        self.from_generator = inspect.isgeneratorfunction(self.layer_op)
        self.shuffle = shuffle
        self.epoch = 1 if self.from_generator else epoch
        self.smaller_final_batch_mode = look_up_operations(
            smaller_final_batch_mode.lower(), SMALLER_FINAL_BATCH_MODE)

        self.n_subjects = 1
        self.window = None
        if reader is not None:
            self.window = ImageWindow.from_data_reader_properties(
                reader.input_sources,
                reader.shapes,
                reader.tf_dtypes,
                window_sizes or (-1, -1, -1))
            self.n_subjects = reader.num_subjects
            self.window.n_samples = windows_per_image

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
        assert self.window, 'Unknown output dtypes: self.window not initialised'
        return self.window.tf_dtypes

    def set_num_threads(self, num_threads):
        """
        Set number windows to generate in parallel.
        """
        self._num_threads = int(num_threads)

    def layer_op(self, idx=None):
        """
        Generating each image as a window.
        Overriding this function to create new image sampling strategies.

        This function should either yield or return a dictionary
        (of multiple windows per image)::

            return a dictionary:
            {
             'image_name': a numpy array [n_samples, h, w, d, chn],
             'image_name_location': [n_samples, 7]
            }

        where the 7-element location vector encode the image_id,
        starting and ending coordinates of the image window.

        Following the same notation, the dictionary can be extended
        to multiple modalities; the keys will be::

            {'image_name_1', 'image_name_1_location',
             'image_name_2', 'image_name_2_location', ...}

        :param idx: image_id used to load the image at the i-th row of
            the input
        :return: a image data dictionary
        """
        image_id, image_data, _ = self.reader(idx=idx)
        for mod in list(image_data):
            spatial_shape = image_data[mod].shape[:N_SPATIAL]
            coords = self.dummy_coordinates(image_id, spatial_shape, 1)
            image_data[LOCATION_FORMAT.format(mod)] = coords
            image_data[mod] = image_data[mod][np.newaxis, ...]
        return image_data

        # # The following is a demo of generator as the layer_op
        # # Often we don't know the total number of elements that
        # # will be generated, epoch is always 1.
        # for idx in range(100):
        #     image_id, image_data, _ = self.reader()
        #     for mod in list(image_data):
        #         spatial_shape = image_data[mod].shape[:N_SPATIAL]
        #         coords = self.dummy_coordinates(image_id, spatial_shape, 1)
        #         image_data[LOCATION_FORMAT.format(mod)] = coords
        #         image_data[mod] = image_data[mod][np.newaxis, ...]
        #     yield image_data

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
            # in case `run_threads` is not called,
            # here we initialise the dataset and iterator
            self.init_dataset()
            self.iterator = self.dataset.make_one_shot_iterator()
            # self.iterator = tf.data.Iterator.from_structure(
            #     self.dataset.output_types, self.dataset.output_shapes)

        window_output = self.iterator.get_next()
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
        dataset = dataset.prefetch(buffer_size=self.queue_length)
        if self.shuffle:
            # locally shuffle the buffer of image windows
            dataset = dataset.shuffle(
                buffer_size=self.queue_length, seed=self._seed)

        if self.smaller_final_batch_mode == 'drop':
            # drop the remainder if there's not enough windows to
            # form a batch, so that we have a fixed batch size.
            # dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(
            #     batch_size=self.batch_size))
            # new API since TF 1.10
            dataset = dataset.batch(batch_size=self.batch_size,
                                    drop_remainder=True)
            return dataset

        dataset = dataset.batch(batch_size=self.batch_size)

        if self.smaller_final_batch_mode == 'dynamic' and self.batch_size > 1:
            return dataset

        # self.smaller_final_batch_mode is 'pad'
        # if self.batch_size == 1 no actual padding
        # but this function will set the output shapes properly.
        def _pad_batch(batch_size):
            def _pad_batch_func(input_tensor_dict):
                """
                function to pad the batch dim to `batch_size`.
                (assuming the input dataset is a dictionary-based one)
                """
                out_dict = {}
                for in_name in list(input_tensor_dict):
                    in_var = input_tensor_dict[in_name]
                    var_shape = in_var.shape.as_list()
                    if batch_size > 1:
                        paddings = [[0, 0] for _ in in_var.shape]
                        paddings[0][1] = batch_size - tf.shape(in_var)[0]
                        in_var = tf.pad(
                            in_var, paddings, "CONSTANT", constant_values=-1)
                    var_shape[0] = batch_size
                    in_var.set_shape(var_shape)
                    out_dict[in_name] = in_var
                return out_dict

            return _pad_batch_func

        dataset = dataset.map(_pad_batch(self.batch_size))
        return dataset

    # pylint: disable=redefined-variable-type
    def _dataset_from_range(self):
        """
        This function maps a dataset of integers to a dataset of images.

        :return: a `tf.data.Dataset`
        """
        # dataset: a list of integers
        tf.logging.info(
            'Initialising Dataset from %s subjects...', self.n_subjects)
        dataset = tf.data.Dataset.range(self.n_subjects)
        if self.shuffle:
            # global shuffle of the entire set of subjects
            dataset = dataset.shuffle(
                buffer_size=self.n_subjects, seed=self._seed)

        # dataset: map each integer i to n windows sampled from subject i
        def _tf_wrapper(idx):
            flattened_types = nest.flatten(self.tf_dtypes)
            flattened_shapes = nest.flatten(self.tf_shapes)
            flat_values = tf.py_func(
                func=lambda subject_id: nest.flatten(self(subject_id)),
                inp=[idx],
                Tout=flattened_types)
            for ret_t, shape in zip(flat_values, flattened_shapes):
                # the actual returned numpy array shapes are not checked
                ret_t.set_shape(shape)
            return nest.pack_sequence_as(self.tf_dtypes, flat_values)

        dataset = dataset.map(_tf_wrapper, num_parallel_calls=self._num_threads)

        # dataset: slice the n-element window into n single windows
        dataset = dataset.flat_map(map_func=tf.data.Dataset.from_tensor_slices)
        return dataset

    def _dataset_from_generator(self):
        """
        Create a `tf.data.Dataset` from a layer_op (as a generator).

        :return: a `tf.data.Dataset`
        """
        tf.logging.info('Initialising dataset from generator...')

        if self._num_threads < 2 or not self.shuffle:
            window_generator = self
        else:
            # self._enqueuer = GeneratorEnqueuer(
            #     self(),
            #     use_multiprocessing=True,
            #     wait_time=0.01,
            #     seed=self._seed)
            self._enqueuer = GeneratorEnqueuer(
                self(),
                use_multiprocessing=True)
            self._enqueuer.start(
                workers=self._num_threads, max_queue_size=self.queue_length)
            window_generator = self._enqueuer.get

        # dataset from generator
        dataset = tf.data.Dataset.from_generator(
            generator=window_generator,
            output_types=self.tf_dtypes,
            output_shapes=self.tf_shapes)

        # dataset: slice the n-element window into n single windows
        dataset = dataset.flat_map(map_func=tf.data.Dataset.from_tensor_slices)
        return dataset

    def run_threads(self, *_args, **_kwargs):
        """
        This function is created for compatibility purposes

        (Deprecating)

        :param _args:
        :param _kwargs:
        :return:
        """
        pass
        # if self.dataset is None or self.iterator is None:
        #     self.init_dataset()
        #     self.iterator = self.dataset.make_one_shot_iterator()

        #     self.iterator = tf.data.Iterator.from_structure(
        #         self.dataset.output_types, self.dataset.output_shapes)
        # sess = session or tf.get_default_session()
        # if sess is not None:
        #     sess.run(self.iterator.make_initializer(self.dataset))

    def close_all(self):
        """
        For compatibility with the queue-based sampler.
        """
        if self._enqueuer is not None:
            self._enqueuer.stop()

    @classmethod
    def dummy_coordinates(cls, image_id, image_sizes, n_samples):
        """
        This function returns a set of image window coordinates
        which are just spatially from 0 to `image_sizes`.

        :return: a numpy array of `n_samples` spatial coordinates
        """

        starting_coordinates = [0, 0, 0]
        image_spatial_shape = list(image_sizes[:N_SPATIAL])
        coords = [image_id] + starting_coordinates + image_spatial_shape
        coords = np.tile(np.asarray(coords), [n_samples, 1])
        return coords.astype(BUFFER_POSITION_NP_TYPE)
