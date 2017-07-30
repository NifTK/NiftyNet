# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from abc import ABCMeta, abstractmethod

import tensorflow as tf

import  niftynet.io.misc_io as util
from niftynet.io.misc_io import infer_ndims_from_file
from niftynet.io.misc_io import load_image


class Loadable(object):
    """
    interface of loadable data
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_as_5d_matrix(self):
        raise NotImplementedError


class DataFromFile(Loadable):
    def __init__(self, file_path, name):
        self.file_path = file_path
        self.name = name
        assert len(self.file_path) == len(self.name), \
            "file_path and modalitiy names are not consistent."

    @property
    def file_path(self):
        assert isinstance(self._file_path, tuple)
        return self._file_path

    @file_path.setter
    def file_path(self, path_array):
        assert path_array is not None, \
            "no file path provided for input volume"
        if isinstance(path_array, basestring):
            self._file_path = (path_array,)
        elif isinstance(path_array, list):
            self._file_path = tuple(path_array)
        elif isinstance(path_array, tuple):
            self._file_path = tuple(path_array)  # pylint: disable=W0201
        else:
            tf.logging.fatal('file_path: prefer a tuple of path strings')
            raise ValueError

        if not all([os.path.isfile(file_name)
                    for file_name in self._file_path]):
            tf.logging.fatal("data file not found".format(self._file_path))
            raise IOError

    @property
    def name(self):
        assert isinstance(self._name, tuple)
        return self._name

    @name.setter
    def name(self, name_array):
        if isinstance(name_array, basestring):
            self._name = (name_array,)
        elif isinstance(name_array, list):
            self._name = tuple(name_array)
        elif isinstance(name_array, tuple):
            self._name = name_array  # pylint: disable=W0201
        else:
            tf.logging.fatal('name: prefer a tuple of name strings')
            raise ValueError

    def load_as_5d_matrix(self):
        print('called data from file')


class SpatialImage2D(DataFromFile):
    def __init__(self, file_path, name, interp_order):
        super(SpatialImage2D, self).__init__(file_path=file_path,
                                             name=name)
        self.interp_order = interp_order
        self._pixdim = None

    @property
    def interp_order(self):
        assert isinstance(self._interp_order, tuple)
        return self._interp_order

    @interp_order.setter
    def interp_order(self, interp_order):
        if isinstance(interp_order, int):
            self._interp_order = (interp_order,)
        elif isinstance(interp_order, list):
            self._interp_order = tuple(interp_order)
        elif isinstance(interp_order, tuple):
            self._interp_order = interp_order  # pylint: disable=W0201
        else:
            tf.logging.fatal('interp_order: prefer a tuple of integers')
            raise ValueError
        assert len(self.interp_order) == len(self.file_path), \
            "length of interp_order and file_path not consistent"

    def load_as_5d_matrix(self):
        print('called spatial image 2d')


class SpatialImage3D(SpatialImage2D):
    def __init__(self, file_path, name, interp_order):
        super(SpatialImage3D, self).__init__(file_path=file_path,
                                             name=name,
                                             interp_order=interp_order)
        self._affine = None
        self.load_header()

    def load_header(self):
        __obj = load_image(self.file_path[0])
        util.correct_image_if_necessary(__obj)
        self._affine = __obj.affine
        self._pixdim = __obj.header.get_zooms()[:3]


    def load_as_5d_matrix(self):
        print('called spatial image 3d')


class SpatialImage4D(SpatialImage3D):
    def __init__(self, file_path, name, interp_order):
        super(SpatialImage4D, self).__init__(file_path=file_path,
                                             name=name,
                                             interp_order=interp_order)

    def load_as_5d_matrix(self):
        SpatialImage3D.load_as_5d_matrix(self)
        print('called spatial image 4d')


class ImageFactory(object):
    @staticmethod
    def create_instance(file_path, **kwargs):

        is_image_from_multi_files = \
            isinstance(file_path, tuple) or isinstance(file_path, list)

        if is_image_from_multi_files:
            multi_mod_dim = 1 if len(file_path) > 1 else 0
            ndims = infer_ndims_from_file(file_path[0]) + multi_mod_dim
        else:
            ndims = infer_ndims_from_file(file_path)

        if ndims == 2:
            image_type = SpatialImage2D
        elif ndims == 3:
            image_type = SpatialImage3D
        elif ndims == 4:
            image_type = SpatialImage4D
        else:
            raise NotImplementedError
        return image_type(file_path, **kwargs)
