# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from abc import ABCMeta, abstractmethod

import nibabel as nib  # TODO: move this to misc
import numpy as np
import tensorflow as tf

import niftynet.io.misc_io as misc
from niftynet.utilities.user_parameters_helper import validate_input_tuple

STD_ORIENTATION = [[0, 1], [1, 1], [2, 1]]
STD_PIXDIM = [1, 1, 1]


class Loadable(object):
    """
    interface of loadable data
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def load_as_5d_matrix(self, *args, **kwargs):
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
        self._file_path = validate_input_tuple(path_array, basestring)
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
        self._name = validate_input_tuple(name_array, basestring)

    def load_as_5d_matrix(self):
        raise NotImplementedError


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
        self._interp_order = validate_input_tuple(interp_order, int)
        assert len(self.interp_order) == len(self.file_path), \
            "length of interp_order and file_path not consistent"

    def load_as_5d_matrix(self, is_resampling=False):
        if len(self._file_path) > 1:
            # 2D image from multiple files
            raise NotImplementedError
        image_obj = misc.load_image(self.file_path[0])
        image_data = image_obj.get_data().astype(np.float32)
        image_data = misc.expand_to_5d(image_data)
        if is_resampling:
            raise NotImplementedError
        return image_data


class SpatialImage3D(SpatialImage2D):
    def __init__(self, file_path, name, interp_order):
        super(SpatialImage3D, self).__init__(file_path=file_path,
                                             name=name,
                                             interp_order=interp_order)
        self._affine = None
        self.load_header()

    def load_header(self):
        __obj = misc.load_image(self.file_path[0])
        misc.correct_image_if_necessary(__obj)
        self._affine = __obj.affine
        # assumes len(pixdims) == 3
        self._pixdim = __obj.header.get_zooms()[:3]

    def load_as_5d_matrix(self, do_resampling=False, do_reorientation=False):
        if len(self._file_path) > 1:
            # 3D image from multiple 2d files
            raise NotImplementedError
        # assuming len(self._file_path) == 1
        image_obj = misc.load_image(self.file_path[0])
        image_data = image_obj.get_data().astype(np.float32)
        image_data = misc.expand_to_5d(image_data)

        if do_resampling and self._pixdim is not None:
            image_data = misc.do_resampling(image_data,
                                            self._pixdim,
                                            STD_PIXDIM,
                                            self._interp_order[0])
        if do_reorientation and self._affine is not None:
            __orientation = nib.orientations.axcodes2ornt(
                nib.aff2axcodes(self._affine))
            image_data = misc.do_reorientation(image_data,
                                               __orientation,
                                               STD_ORIENTATION)
        return image_data


class SpatialImage4D(SpatialImage3D):
    def __init__(self, file_path, name, interp_order):
        super(SpatialImage4D, self).__init__(file_path=file_path,
                                             name=name,
                                             interp_order=interp_order)

    def load_as_5d_matrix(self, do_resampling=False, do_reorientation=False):
        if len(self._file_path) == 1:
            # 4D image from a single file
            raise NotImplementedError
        # assuming len(self._file_path) > 1
        mod_list = []
        for mod in range(len(self._file_path)):
            __file_path = (self.file_path[mod],)
            __name = (self.name[mod],)
            __interp_order = (self._interp_order[mod],)

            mod_3d = SpatialImage3D(file_path=__file_path,
                                    name=__name,
                                    interp_order=__interp_order)
            mod_data_5d = mod_3d.load_as_5d_matrix(do_resampling,
                                                   do_reorientation)
            mod_list.append(mod_data_5d)
        try:
            image_data = np.concatenate(mod_list, axis=3)
        except ValueError:
            tf.logging.fatal(
                "multi-modal data shapes not consistent -- currently"
                "assumes same-shape 3D volumes, pixdim, and affines")
            raise
        return image_data


class ImageFactory(object):
    INSTANCE_DICT = {2: SpatialImage2D,
                     3: SpatialImage3D,
                     4: SpatialImage4D}

    @classmethod
    def create_instance(cls, file_path, **kwargs):

        is_image_from_multi_files = \
            isinstance(file_path, tuple) or isinstance(file_path, list)

        if is_image_from_multi_files:
            multi_mod_dim = 1 if len(file_path) > 1 else 0
            ndims = misc.infer_ndims_from_file(file_path[0]) + multi_mod_dim
        else:
            ndims = misc.infer_ndims_from_file(file_path)

        image_type = cls.INSTANCE_DICT.get(ndims, None)
        if image_type is None:
            raise NotImplementedError
        return image_type(file_path, **kwargs)
