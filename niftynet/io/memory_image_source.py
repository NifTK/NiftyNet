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

            for name in self._section_names:
                funct \
                    = self._input_callback_functions[self._modality_names[name]]
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
