# -*- coding: utf-8 -*-
"""
This module provides the factory for image data-set
partitioners, and image sources and sinks
"""
from __future__ import absolute_import

from niftynet.io.file_image_sets_partitioner import FileImageSetsPartitioner
from niftynet.io.file_image_sink import FileImageSink
from niftynet.io.file_image_source import DEFAULT_INTERP_ORDER
from niftynet.io.image_reader import ImageReader
from niftynet.io.memory_image_sets_partitioner import (
    MemoryImageSetsPartitioner, is_memory_data_param)
from niftynet.io.memory_image_sink import (MEMORY_OUTPUT_CALLBACK_PARAM,
                                           MemoryImageSink)
from niftynet.utilities.decorators import singleton

ENDPOINT_MEMORY = 'memory'
ENDPOINT_FILESYSTEM = 'filesystem'
# List of supported endpoints
SUPPORTED_ENDPOINTS = (ENDPOINT_FILESYSTEM, ENDPOINT_MEMORY)


@singleton
class ImageEndPointFactory(object):
    """
    This singleton class allows for the instantiation
    and configuration of any image source and/or data-set
    partitioner
    """

    _data_param = None
    _task_param = None
    _action_param = None
    _endpoint_type = ENDPOINT_FILESYSTEM
    _sink_classes = {
        ENDPOINT_FILESYSTEM: FileImageSink,
        ENDPOINT_MEMORY: MemoryImageSink
    }
    _partitioner_classes = {
        ENDPOINT_FILESYSTEM: FileImageSetsPartitioner,
        ENDPOINT_MEMORY: MemoryImageSetsPartitioner
    }
    _partitioner = None

    def set_params(self, data_param, task_param, action_param):
        """
        Configures this factory and determines which type of
        image endpoints and partitioners are instantiated.

        :param data_param: Data specification
        :param task_param: Application task specification
        """

        self._data_param = data_param
        self._task_param = task_param
        self._action_param = action_param

        if data_param is not None \
           and is_memory_data_param(data_param):
            self._endpoint_type = ENDPOINT_MEMORY
        else:
            self._endpoint_type = ENDPOINT_FILESYSTEM

    def create_partitioner(self):
        """
        Instantiates a new data-set partitioner suitable for the image
        end-point type specified via this factory's parameters.
        """

        if self._data_param is None and self._task_param is None:
            raise RuntimeError('Application parameters must be set before any'
                               ' data set can be partitioned.')

        assert self._endpoint_type == ENDPOINT_MEMORY \
            or not is_memory_data_param(self._data_param)

        self._partitioner = self._partitioner_classes[self._endpoint_type]()

        return self._partitioner

    def create_sources(self, dataset_names, phase, action):
        """
        Instantiates a list of sources for the specified application phase
        and data-set names.
        :param phase: an application life-cycle phase, e.g, TRAIN, VALID
        :param action: application action, e.g., TRAIN, INFER
        :param dataset_names: image collection/modality names
        :return: a configured image source
        """

        if self._partitioner is None:
            raise RuntimeError('Sources can only be instantiated after'
                               'data set partitioning')

        _sources = []
        for subject_list in self._partitioner.get_image_lists_by(
                phase=phase, action=action):
            if self._endpoint_type == ENDPOINT_FILESYSTEM:
                _source = ImageReader(dataset_names).initialise(
                    self._data_param,
                    self._task_param,
                    file_list=subject_list,
                    from_files=True)
            else:
                _source = ImageReader(dataset_names).initialise(
                    self._data_param,
                    self._task_param,
                    phase_indices=subject_list,
                    from_files=False)
            _sources.append(_source)
        return _sources

    def create_sinks(self, sources):
        """
        Produces and configures the image sinks used for inference result
        output.
        """

        if 'output_interp_order' in vars(self._action_param):
            interp_order = self._action_param.output_interp_order
        else:
            interp_order = DEFAULT_INTERP_ORDER

        if self._endpoint_type == ENDPOINT_FILESYSTEM:
            kwargs = {}
            if 'save_seg_dir' in vars(self._action_param):
                kwargs['output_path'] = self._action_param.save_seg_dir

            if 'output_postfix' in vars(self._action_param):
                kwargs['postfix'] = self._action_param.output_postfix

            return [
                self._sink_classes[ENDPOINT_FILESYSTEM](
                    sources[0] if sources else None, interp_order, **kwargs)
            ]

        if MEMORY_OUTPUT_CALLBACK_PARAM not in vars(self._action_param):
            raise RuntimeError(
                'In memory I/O mode an output callback '
                'function must be provided via the field'
                ' %s in action_param' % MEMORY_OUTPUT_CALLBACK_PARAM)

        return [
            self._sink_classes[ENDPOINT_MEMORY](
                sources[0], self._action_param.output_interp_order,
                vars(self._action_param)[MEMORY_OUTPUT_CALLBACK_PARAM])
        ]
