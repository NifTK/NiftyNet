# -*- coding: utf-8 -*-
"""
In-memory image passing support module
"""
from __future__ import absolute_import

from copy import deepcopy

import numpy as np
import tensorflow as tf

from niftynet.io.base_image_source import DEFAULT_INTERP_ORDER, BaseImageSource
from niftynet.io.misc_io import dtype_casting, expand_to_5d

# Name of the data_param namespace entry for the memory input sources
MEMORY_INPUT_CALLBACK_PARAM = 'input_callback'


class MemoryImageSource(BaseImageSource):
    """
    This class acts as a compatibility layer between a callback
    function yielding ID-data tuples and code expecting an ImageReader
    layer.
    """

    def __init__(self, name='memory_image_reader'):
        super(MemoryImageSource, self).__init__(name=name)

        self._input_callback_functions = None
        self._modality_interp_orders = None
        self._phase_indices = None

    def initialise(self, data_param, input_sources, phase_indices=None):
        """

        :param data_param: data specification dict
        :param input_sources: application input modality specification dict
        :param phase_indices: subset of image indices to consider in this phase

        :return: self
        """
        self._input_sources = input_sources

        # read required input callback for each modality
        required_modalities = [
            list(input_sources.get(name)) for name in input_sources
        ]
        required_modalities = set(sum(required_modalities, []))
        self._input_callback_functions = {}
        for name in required_modalities:
            try:
                self._input_callback_functions[name] = \
                    data_param[name][MEMORY_INPUT_CALLBACK_PARAM]
            except (AttributeError, TypeError, ValueError):
                tf.logging.fatal('Require an input callback for modality %s',
                                 name)
                raise
        # set interpolation order based on `data_param['interp_order']`
        self._modality_interp_orders = \
            {name: tuple(
                [data_param[mod].get('interp_order', DEFAULT_INTERP_ORDER)
                 for mod in mods])
             for name, mods in self.input_sources.items()}
        self._phase_indices = phase_indices
        return self

    def __assemble_output(self, idx, source_name):
        """
        Assembles the output image for the given source
        by stacking its modalities
        """

        if not self.input_sources:
            tf.logging.fatal('This source is not initialised.')
            raise RuntimeError

        image_idx = self._phase_indices[idx]
        modalities = self.input_sources[source_name]

        image_data = [
            self._input_callback_functions[mod](image_idx)
            for mod in modalities
        ]
        try:
            return np.concatenate(image_data, axis=-1)
        except ValueError:
            tf.logging.fatal('Only images with identical spatial '
                             'configuration can be stacked. Please '
                             'adapt your callback functions.')
            raise

    def __extract_image_property(self, property_function):
        """
        Extracts a property of the output images by means of an
        argument function

        :param property_function: function that returns a sought
            image property given an image
        """
        return {
            name: property_function(self.__assemble_output(0, name), name)
            for name in self.input_sources
        }

    def _load_spatial_ranks(self):
        def __rank_func(img, _unused_name):
            return 3 if img.shape[2] > 1 else 2

        return self.__extract_image_property(__rank_func)

    def _load_shapes(self):
        def __shape_func(img, _unused_name):
            return img.shape

        return self.__extract_image_property(__shape_func)

    def _load_dtypes(self):
        def __dtype_func(img, name):
            return dtype_casting(
                img.dtype, self._modality_interp_orders[name][0], as_tf=True)

        return self.__extract_image_property(__dtype_func)

    @property
    def num_subjects(self):
        return len(self._phase_indices) if self._phase_indices else -1

    def get_image_index(self, subject_id):
        if not self._phase_indices:
            return -1
        idx = np.argwhere(
            np.array(self._phase_indices).reshape(-1) == int(subject_id))
        return idx[0, 0] if idx else -1

    def get_subject_id(self, image_index):
        return '{}'.format(self._phase_indices[image_index])

    def get_image(self, idx):
        return {
            name: self.__assemble_output(idx, name)
            for name in self.input_sources
        }

    def get_image_and_interp_dict(self, idx):
        try:
            return self.get_image(idx), deepcopy(self._modality_interp_orders)
        except (TypeError, IndexError):
            return None, None


# pylint: disable=too-many-arguments
def make_input_spec(modality_spec,
                    image_callback_function,
                    do_reshape_nd=False,
                    do_reshape_rgb=False,
                    do_typecast=True,
                    additional_callbacks=None):
    """
    Updates a configuration-file modality specification with the
    necessary fields for loading from memory.

    :param image_callback_function: the function yielding the image tensor
        given an index.
    :param modality_spec: the original specification of the modality,
        containing window sizes, pixel dimensions, etc.
    :param do_reshape_nd: boolean flag indicating whether to add an image
        tensor reshape wrapper to the function turning it a nD input tensor
        (with possibly multiple modalities) into a 5D one, as required by
        NiftyNet, while interpretting the last dimension as modalities.
    :param do_reshape_rgb: boolean flag indicating whether to add an image
        tensor reshape wrapper to the function turning it a 2D RGB image
        into a 5D tensor, as required by NiftyNet.
    :param do_typecast: boolean flag indicating whether to add an image
        data typecast wrapper to the function, turning the data into floats.
    :param additional_callbacks: a list of function transforms which
        convert a numpy array into another numpy array, as preprocessors
    """

    def _reshape_wrapper_nd(img):
        img = expand_to_5d(img)
        if img.shape[3] > 1:
            # time sequences (4th dim length > 1) not supported
            # all content squeezed to the 5th dim (modality dim)
            new_shape = img.shape[0:3] + (1, -1)
            img = img.reshape(new_shape)
        return img

    def _reshape_wrapper_rgb(img):
        new_shape = img.shape[0:2] + (1, 1, 3)
        return img.reshape(new_shape)

    def _typecast_wrapper(img):
        return img.astype(np.float32)

    img_callbacks = []
    if do_reshape_nd:
        img_callbacks.append(_reshape_wrapper_nd)
    if do_reshape_rgb:
        img_callbacks.append(_reshape_wrapper_rgb)
    if do_typecast:
        img_callbacks.append(_typecast_wrapper)
    if additional_callbacks:
        img_callbacks.extend(additional_callbacks)

    def _stacked_call(idx):
        img = image_callback_function(idx)
        for func in img_callbacks:
            img = func(img)
        return img

    if not isinstance(modality_spec, dict):
        vars(modality_spec)[MEMORY_INPUT_CALLBACK_PARAM] = _stacked_call
    else:
        modality_spec[MEMORY_INPUT_CALLBACK_PARAM] = _stacked_call
    return modality_spec
