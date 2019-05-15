# -*- coding: utf-8 -*-
"""
In-memory image passing support module
"""
from __future__ import absolute_import

from copy import deepcopy

import numpy as np

from niftynet.io.base_image_source import BaseImageSource

# Name of the data_param namespace entry for the memory input sources
MEMORY_INPUT_CALLBACK_PARAM = 'input_callback'


class MemoryImageSource(BaseImageSource):
    """
    This class acts as a compatibility layer between a callback
    function yielding ID-data tuples and code expecting an ImageReader
    layer.
    """

    def __init__(self, section_names, name='memory_image_source'):
        """
        :param section_names: the list of section names (in
        task_param) describing the modalities.
        """

        super(MemoryImageSource, self).__init__()

        self._input_callback_functions = None
        self._modality_interp_orders = None
        self._phase_indices = None
        self._section_names = section_names
        self._modality_names = None

    def initialise(self, data_param, task_param, phase_indices):
        """
        :param data_param: Data specification
        :param task_param: Application task specification
        :param phase_indices: subset of image indices to consider in this phase
        :return: self
        """

        self._section_names, self._modality_names \
            = self._get_section_input_sources(task_param, self._section_names)

        self._input_callback_functions = {}
        all_modalities = []
        for mods in self._modality_names.values():
            all_modalities += mods
        for name in set(all_modalities):
            if not vars(data_param[name]).get(MEMORY_INPUT_CALLBACK_PARAM,
                                              None):
                raise ValueError(
                    'Require an input callback for modality %s' % name)

            self._input_callback_functions[name] \
                = vars(data_param[name])[MEMORY_INPUT_CALLBACK_PARAM]

        self._modality_interp_orders \
            = {name: tuple([data_param[mod].interp_order for mod in mods])
               for name, mods in self._modality_names.items()}
        self._phase_indices = phase_indices

        return self

    def _assemble_output(self, idx, source_name):
        """
        Assembles the output image for the given source
        by stacking its modalities
        """

        if not self._modality_names:
            raise RuntimeError('This source is not initialised.')

        image_idx = self._phase_indices[idx]
        modalities = self._modality_names[source_name]
        first_image = self._input_callback_functions[modalities[0]](image_idx)

        if len(modalities) > 1:
            images = [first_image]
            for mod in modalities[1:]:
                images.append(self._input_callback_functions[mod](image_idx))
                if first_image.shape[:3] != images[-1].shape[:3]:
                    raise RuntimeError('Only images with identical spatial '
                                       'configuration can be stacked. Please '
                                       'adapt your callback functions.')

            return np.concatenate(images, axis=-1)

        return first_image

    def get_output_image(self, idx):
        return {
            name: self._assemble_output(0, name)
            for name in self.names
        }

    @property
    def names(self):
        return list(self._modality_names.keys())

    @property
    def num_subjects(self):
        return len(self._phase_indices)

    @property
    def input_sources(self):
        return self._modality_names

    def _extract_image_property(self, property_function):
        """
        Extracts a property of the output images by means of an
        argument function

        :param property_function: function that returns a sought
            image property given an image
        """
        return {
            name: property_function(self._assemble_output(0, name))
            for name in self._modality_names
        }

    def _load_spatial_ranks(self):
        return self._extract_image_property(
            lambda img: 3 if img.shape[2] > 1 else 2)

    def _load_shapes(self):
        return self._extract_image_property(lambda img: img.shape)

    def _load_dtypes(self):
        return self._extract_image_property(lambda img: img.dtype)

    def get_image_index(self, subject_id):
        idx = np.argwhere(np.array(self._phase_indices) == int(subject_id))

        return idx[0, 0] if idx else -1

    def get_subject_id(self, image_index):
        return str(self._phase_indices[image_index])

    def _get_image_and_interp_dict(self, idx):
        try:
            image_data = {}

            for name in self._section_names:
                data = self._assemble_output(idx, name)
                image_data[name] = data

            return image_data, deepcopy(self._modality_interp_orders)
        except (TypeError, IndexError):
            return None, None


def make_input_spec(modality_spec, image_callback_function, do_reshape_nd=False,
                    do_reshape_rgb=False, do_typecast=True):
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
    """

    callback0 = image_callback_function
    if do_reshape_nd:
        def _reshape_wrapper_nd(idx):
            img = callback0(idx)

            new_shape = list(img.shape[:3])
            new_shape += [1]*(4 - len(new_shape))
            if len(img.shape) > 3:
                new_shape += [img.shape[-1]]
            else:
                new_shape += [1]

            return img.reshape(new_shape)

        callback1 = _reshape_wrapper_nd
    else:
        callback1 = callback0

    if do_reshape_rgb:
        def _reshape_wrapper_rgb(idx):
            img = callback1(idx)

            return img.reshape((img.shape[0], img.shape[1], 1, 1, 3))

        callback2 = _reshape_wrapper_rgb
    else:
        callback2 = callback1

    if do_typecast:
        def _typecast_wrapper(idx):
            return callback2(idx).astype(np.float32)

        callback3 = _typecast_wrapper
    else:
        callback3 = callback2

    if not isinstance(modality_spec, dict):
        vars(modality_spec)[MEMORY_INPUT_CALLBACK_PARAM] = callback3
    else:
        modality_spec[MEMORY_INPUT_CALLBACK_PARAM] = callback3

    return modality_spec
