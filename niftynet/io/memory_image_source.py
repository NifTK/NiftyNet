# -*- coding: utf-8 -*-
"""
In-memory image passing support module
"""
from __future__ import absolute_import

from copy import deepcopy

import numpy as np
from niftynet.io.base_image_source import BaseImageSource, infer_tf_dtypes
from niftynet.io.image_loader import image2nibabel

# Name of the data_param namespace entry for the memory input sources
MEMORY_INPUT_CALLBACK_PARAM = 'input_callback'


class MemoryImageSource(BaseImageSource):
    """
    This class acts as a compatibility layer between a callback
    function yielding ID-data tuples and code expecting an ImageReader
    layer.
    """

    def __init__(self, modality_names, name='memory_image_source'):
        """
        :param modality_names: the list of modality names (in
        data_param) to read from.
        """

        super(MemoryImageSource, self).__init__()

        self._input_callback_functions = None
        self._modality_interp_orders = None
        self._phase_indices = None
        self._modality_names = modality_names

    # pylint: disable=unused-argument
    def initialise(self, data_param, task_param, phase_indices):
        """
        :param data_param: Data specification
        :param task_param: Application task specification
        :param phase_indices: subset of image indices to consider in this phase
        :return: self
        """

        self._input_callback_functions \
            = {name: vars(data_param[name])[MEMORY_INPUT_CALLBACK_PARAM]
               for name in self._modality_names}
        self._modality_interp_orders \
            = {name: data_param[name].interp_order
               for name in self._modality_names}
        self._phase_indices = phase_indices

        return self

    @property
    def names(self):
        return list(self._input_callback_functions.keys())

    @property
    def num_subjects(self):
        return len(self._phase_indices)

    def _load_spatial_ranks(self):
        return {
            name: source(0).spatial_rank
            for name, source in self._input_callback_functions.items()
        }

    def _load_shapes(self):
        return {
            name: source(0).shape
            for name, source in self._input_callback_functions.items()
        }

    def _load_dtypes(self):
        return {
            name: infer_tf_dtypes(source(0))
            for name, source in self._input_callback_functions.items()
        }

    def get_image_index(self, subject_id):
        idx = np.argwhere(np.array(self._phase_indices) == int(subject_id))

        return idx[0, 0] if idx else -1

    def get_subject_id(self, image_index):
        return str(self._phase_indices[image_index])

    def _get_image_and_interp_dict(self, idx):
        try:
            image_data = {}

            for name, funct in self._input_callback_functions.items():
                data = funct(self._phase_indices[idx]).get_data()
                image_data[name] = data

            return image_data, deepcopy(self._modality_interp_orders)
        except (TypeError, IndexError):
            return None, None


def make_input_spec(modality_spec, image_callback_function):
    """
    Updates a configuration-file modality specification with the
    necessary fields for loading from memory.

    :param image_callback_function: the function yielding the image tensor
        given an index.
    :param modality_spec: the original specification of the modality,
        containing window sizes, pixel dimensions, etc.
    """

    def _image_output_wrapper(idx):
        return image2nibabel(image_callback_function(idx))

    if isinstance(modality_spec, dict):
        modality_spec[MEMORY_INPUT_CALLBACK_PARAM] = _image_output_wrapper
    else:
        vars(modality_spec)[MEMORY_INPUT_CALLBACK_PARAM] = \
            _image_output_wrapper

    return modality_spec
