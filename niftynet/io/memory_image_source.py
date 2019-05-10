# -*- coding: utf-8 -*-
"""
In-memory image passing support module
"""
from __future__ import absolute_import

from copy import deepcopy

import numpy as np
from niftynet.io.base_image_source import BaseImageSource, infer_tf_dtypes
from niftynet.io.misc_io import dtype_casting

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

        if any(len(mods) != 1 for mods in self._modality_names.values()):
            raise ValueError('Memory I/O supports only 1 modality'
                             ' per application image section. Please'
                             ' stack your modalities prior to passing '
                             'them to the callback function.')

        self._modality_names = {name: mods[0] for name, mods
                                in self._modality_names.items()}

        self._input_callback_functions = {}
        for name in self._modality_names.values():
            if MEMORY_INPUT_CALLBACK_PARAM not in vars(data_param[name]) \
                or vars(data_param[name])[MEMORY_INPUT_CALLBACK_PARAM] is None:
                raise ValueError('Require an input callback for modality %s'
                                 % name)

            self._input_callback_functions[name] \
                = vars(data_param[name])[MEMORY_INPUT_CALLBACK_PARAM]

        self._modality_interp_orders \
            = {name: (data_param[mod].interp_order,)
               for name, mod in self._modality_names.items()}
        self._phase_indices = phase_indices

        return self

    @property
    def output_list(self):
        return [{name: self._input_callback_functions[mod](idx)
                 for name, mod in self._modality_names.items()}
                for idx in self._phase_indices]

    @property
    def names(self):
        return list(self._input_callback_functions.keys())

    @property
    def num_subjects(self):
        return len(self._phase_indices)

    @property
    def input_sources(self):
        return {name: (mod,) for name, mod in self._modality_names.items()}

    def _load_spatial_ranks(self):
        return {
            name: 3 if self._input_callback_functions[mod](0).shape[2] > 1
            else 2
            for name, mod in self._modality_names.items()
        }

    def _load_shapes(self):
        return {
            name: self._input_callback_functions[mod](0).shape
            for name, mod in self._modality_names.items()
        }

    def _load_dtypes(self):
        return {
            name: dtype_casting(
                self._input_callback_functions[mod](0).dtype,
                self._modality_interp_orders[name][0])
            for name, mod in self._modality_names.items()
        }

    def get_image_index(self, subject_id):
        idx = np.argwhere(np.array(self._phase_indices) == int(subject_id))

        return idx[0, 0] if idx else -1

    def get_subject_id(self, image_index):
        return str(self._phase_indices[image_index])

    def _get_image_and_interp_dict(self, idx):
        try:
            image_data = {}

            for name in self._section_names:
                funct \
                    = self._input_callback_functions[self._modality_names[name]]
                data = funct(self._phase_indices[idx])
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
        into a 5D tensor, as required by
        NiftyNet.
    :param do_typecast: boolean flag indicating whether to add an image
        data typecast wrapper to the function turning it the data into floats.
    """

    callback0 = image_callback_function
    if do_reshape_nd:
        def _reshape_wrapper_nd(idx):
            img = callback0(idx)

            new_shape = list(img.shape)
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
        def _typecase_wrapper(idx):
            return callback2(idx).astype(np.float32)

        callback3 = _typecase_wrapper
    else:
        callback3 = callback2

    if not isinstance(modality_spec, dict):
        vars(modality_spec)[MEMORY_INPUT_CALLBACK_PARAM] = callback3
    else:
        modality_spec[MEMORY_INPUT_CALLBACK_PARAM] = callback3

    return modality_spec
