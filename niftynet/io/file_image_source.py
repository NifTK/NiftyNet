# -*- coding: utf-8 -*-
""" Image F/S output module """

from __future__ import absolute_import

import numpy as np
import pandas
import tensorflow as tf

from niftynet.io.base_image_source import (DEFAULT_INTERP_ORDER,
                                           BaseImageSource, infer_tf_dtypes)
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.io.image_sets_partitioner import COLUMN_UNIQ_ID
from niftynet.io.image_type import ImageFactory
from niftynet.utilities.util_common import print_progress_bar


class FileImageSource(BaseImageSource):
    """
    For a concrete example::

        _input_sources define multiple modality mappings, e.g.,
        _input_sources {'image': ('T1', 'T2'), 'label': ('manual_map',)}

    means:

    'image' consists of two components, formed by
    concatenating 'T1' and 'T2' input source images.
    'label' consists of one component, loading from 'manual_map'

    :param self._input_sources: a dict of the output names of this reader.
        ``{'image': ('T1', 'T2'), 'label': ('manual_map',)}``

    :param self._shapes: the shapes after combining input sources
        ``{'image': (192, 160, 192, 1, 2), 'label': (192, 160, 192, 1, 1)}``

    :param self._dtypes: store the dictionary of tensorflow shapes
        ``{'image': tf.float32, 'label': tf.float32}``

    :param self._output_list: a list of dictionaries, with each item::

        {'image': <niftynet.io.image_type.SpatialImage4D object>,
        'label': <niftynet.io.image_type.SpatialImage3D object>}

    """

    def __init__(self, name='file_image_reader'):
        super(FileImageSource, self).__init__(name=name)

        self._file_list = None  # list of file full names
        self._output_list = None  # list of loadable image instances

    def initialise(self, data_param, input_sources, file_list=None):
        """
        Check if the config file has the required data sources,
        initialise lists of loadable image instances according to the config.
        this function sets the ``file_list`` and ``output_list``
        """
        self._input_sources = input_sources
        if file_list is None:
            # defaulting to all files detected by the input specification
            file_list = \
                ImageSetsPartitioner(data_param).initialise().all_files

        required_modalities = [
            list(input_sources.get(name)) for name in input_sources
        ]
        required_modalities = set(sum(required_modalities, []))
        for required in required_modalities:
            try:
                if (file_list is None) or \
                        (required not in list(file_list)) or \
                        (file_list[required].isnull().all()):
                    tf.logging.fatal(
                        'Reader required input section '
                        'name [%s], but in the filename list '
                        'the column is empty.', required)
                    raise ValueError
            except (AttributeError, TypeError, ValueError):
                tf.logging.fatal(
                    'file_list parameter should be a '
                    'pandas.DataFrame instance and has input '
                    'section name [%s] as a column name.', required)
                if required_modalities:
                    tf.logging.fatal('Reader requires section(s): %s',
                                     required_modalities)
                if file_list is not None:
                    tf.logging.fatal('Configuration input sections are: %s',
                                     list(file_list))
                raise

        self._output_list, self._file_list = _filename_to_image_list(
            file_list, self.input_sources, data_param)
        for name in self.input_sources:
            tf.logging.info(
                'Image reader: loading %d subjects '
                'from sections %s as input [%s]', len(self._output_list),
                self.input_sources[name], name)
        return self

    def _check_initialised(self):
        """
        Checks if the reader has been initialised, if not raises a RuntimeError
        """

        if not self._output_list:
            tf.logging.fatal("Please initialise the reader first.")
            raise RuntimeError

    def _load_spatial_ranks(self):
        self._check_initialised()

        first_image = self._output_list[0]

        return {
            field: first_image[field].spatial_rank
            for field in self.input_sources
        }

    def _load_shapes(self):
        self._check_initialised()

        first_image = self._output_list[0]
        return {
            field: first_image[field].shape
            for field in self.input_sources
        }

    def _load_dtypes(self):
        """
        Infer input data dtypes in TF
        (using the first image in the file list).
        """
        self._check_initialised()

        first_image = self._output_list[0]
        return {
            field: infer_tf_dtypes(first_image[field])
            for field in self.input_sources
        }

    @property
    def output_list(self):
        return self._output_list

    @property
    def num_subjects(self):
        """

        :return: number of subjects in the reader
        """
        return 0 if not self._output_list else len(self._output_list)

    def get_image_index(self, subject_id):
        """
        Given a subject id, return the file_list index
        :param subject_id: a string with the subject id
        :return: an int with the file list index
        """
        return np.flatnonzero(self._file_list['subject_id'] == subject_id)[0]

    def get_subject_id(self, image_index):
        """
        Given an integer id returns the subject id.
        """
        try:
            return self._file_list.iloc[image_index][COLUMN_UNIQ_ID]
        except KeyError:
            tf.logging.warning('Unknown subject id in reader file list.')
            raise

    def get_subject(self, image_index=None):
        """
        Given an integer id returns the corresponding row of the file list.
        returns: a dictionary of the row
        """
        try:
            if image_index is None:
                return self._file_list.iloc[:].to_dict()
            return self._file_list.iloc[image_index].to_dict()
        except (KeyError, AttributeError):
            tf.logging.warning('Unknown subject id in reader file list.')
            raise

    def get_image(self, idx):
        return self._output_list[idx]

    def get_image_and_interp_dict(self, idx):
        try:
            image_data_dict = {
                field: image.get_data()
                for (field, image) in self.get_image(idx).items()
            }
            interp_order_dict = {
                field: image.interp_order
                for (field, image) in self.get_image(idx).items()
            }
            return image_data_dict, interp_order_dict
        except (IndexError, TypeError):
            return None, None


def _filename_to_image_list(file_list, mod_dict, data_param):
    """
    Converting a list of filenames to a list of image objects,
    Properties (e.g. interp_order) are added to each object
    """
    volume_list = []
    valid_idx = []
    for idx in range(len(file_list)):
        # create image instance for each subject
        print_progress_bar(
            idx,
            len(file_list),
            prefix='reading datasets headers',
            decimals=1,
            length=10,
            fill='*')

        # combine fieldnames and volumes as a dictionary
        _dict = {}
        for field, modalities in mod_dict.items():
            _dict[field] = _create_image(file_list, idx, modalities,
                                         data_param)

        # skipping the subject if there're missing image components
        if _dict and None not in list(_dict.values()):
            volume_list.append(_dict)
            valid_idx.append(idx)

    if not volume_list:
        tf.logging.fatal(
            "Empty filename lists, please check the csv "
            "files. (removing csv_file keyword if it is in the config file "
            "to automatically search folders and generate new csv "
            "files again)\n\n"
            "Please note in the matched file names, each subject id are "
            "created by removing all keywords listed `filename_contains` "
            "in the config.\n\n"
            "E.g., `filename_contains=foo, bar` will match file "
            "foo_subject42_bar.nii.gz, and the subject id is _subject42_.")
        raise IOError
    return volume_list, file_list.iloc[valid_idx]


def _create_image(file_list, idx, modalities, data_param):
    """
    data_param consists of description of each modality
    This function combines modalities according to the 'modalities'
    parameter and create <niftynet.io.input_type.SpatialImage*D>
    """
    try:
        file_path = tuple(file_list.iloc[idx][mod] for mod in modalities)
        any_missing = any([
            pandas.isnull(file_name) or not bool(file_name)
            for file_name in file_path
        ])
        if any_missing:
            # to-do: enable missing modalities again
            # the file_path of a multimodal image will contain `nan`, e.g.
            # this should be handled by `ImageFactory.create_instance`
            # ('testT1.nii.gz', 'testT2.nii.gz', nan, 'testFlair.nii.gz')
            return None

        interp_order, pixdim, axcodes, loader = [], [], [], []
        for mod in modalities:
            mod_spec = data_param[mod] \
                if isinstance(data_param[mod], dict) else vars(data_param[mod])
            interp_order.append(
                mod_spec.get('interp_order', DEFAULT_INTERP_ORDER))
            pixdim.append(mod_spec.get('pixdim', None))
            axcodes.append(mod_spec.get('axcodes', None))
            loader.append(mod_spec.get('loader', None))

    except KeyError:
        tf.logging.fatal(
            "Specified modality names %s "
            "not found in config: input sections %s.", modalities,
            list(data_param))
        raise
    except AttributeError:
        tf.logging.fatal(
            "Data params must contain: interp_order, pixdim, axcodes.\n"
            "Reader must be initialised with a dataframe as file_list.")
        raise

    image_properties = {
        'file_path': file_path,
        'name': modalities,
        'interp_order': interp_order,
        'output_pixdim': pixdim,
        'output_axcodes': axcodes,
        'loader': loader
    }
    return ImageFactory.create_instance(**image_properties)
