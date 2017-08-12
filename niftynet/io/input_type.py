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
from niftynet.utilities.user_parameters_helper import make_input_tuple


class Loadable(object):
    """
    interface of loadable data
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_data(self, *args, **kwargs):
        # this function loads a numpy array from the image object
        # if the array has less than 5 dimensions
        # it extends the array to 5d
        # (corresponding to 3 sptial dimensions, temporal dim, modalities)
        # if ndims>5 it should return without modifying the array
        raise NotImplementedError


class DataFromFile(Loadable):
    def __init__(self, file_path, name):
        self.file_path = file_path
        self.name = name
        assert len(self.file_path) == len(self.name), \
            "file_path and modalitiy names are not consistent."
        self._dtype = ()

    @property
    def dtype(self):
        if not self._dtype:
            self._dtype = tuple(misc.load_image(_file).header.get_data_dtype()
                                for _file in self.file_path)
        return self._dtype

    @property
    def file_path(self):
        assert isinstance(self._file_path, tuple)
        return self._file_path

    @file_path.setter
    def file_path(self, path_array):
        self._file_path = make_input_tuple(path_array, basestring)
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
        self._name = make_input_tuple(name_array, basestring)

    def get_data(self):
        raise NotImplementedError


class SpatialImage2D(DataFromFile):
    def __init__(self,
                 file_path,
                 name,
                 interp_order,
                 output_pixdim,
                 **kwargs):
        super(SpatialImage2D, self).__init__(file_path=file_path,
                                             name=name)
        self.interp_order = interp_order
        self.output_pixdim = output_pixdim
        self._original_pixdim = ()
        self._original_shape = ()

    @property
    def shape(self):
        if not self._original_shape:
            self._original_shape = tuple(
                    misc.load_image(_file).header['dim'][1:6]
                    for _file in self.file_path)
            non_modality_shapes = set([tuple(shape[:4].tolist())
                                       for shape in self._original_shape])
            assert len(non_modality_shapes) == 1, \
                "combining multimodal images: shapes not consistent"
            n_modalities = len(self.file_path)
            self._original_shape = non_modality_shapes.pop() + (n_modalities,)
        return self._original_shape

    @property
    def interp_order(self):
        assert isinstance(self._interp_order, tuple)
        return self._interp_order

    @interp_order.setter
    def interp_order(self, interp_order):
        self._interp_order = make_input_tuple(interp_order, int)
        assert len(self._interp_order) == len(self.file_path), \
            "length of interp_order and file_path not consistent"

    @property
    def output_pixdim(self):
        assert isinstance(self._output_pixdim, tuple)
        return self._output_pixdim

    @output_pixdim.setter
    def output_pixdim(self, output_pixdim):
        self._output_pixdim = output_pixdim
        assert len(self._output_pixdim) == len(self.file_path), \
            "length of output_pixdim and file_path not consistent"

    def get_data(self):
        if len(self._file_path) > 1:
            # 2D image from multiple files
            raise NotImplementedError
        image_obj = misc.load_image(self.file_path[0])
        image_data = image_obj.get_data()
        image_data = misc.expand_to_5d(image_data)
        if self._output_pixdim[0]:
            raise NotImplementedError("resampling 2D images not implemented")
        return image_data


class SpatialImage3D(SpatialImage2D):
    def __init__(self,
                 file_path,
                 name,
                 interp_order,
                 output_pixdim,
                 output_axcodes,
                 **kwargs):
        super(SpatialImage3D, self).__init__(file_path=file_path,
                                             name=name,
                                             interp_order=interp_order,
                                             output_pixdim=output_pixdim)
        self.output_axcodes = output_axcodes
        self._affine = None
        self.load_header()

    @property
    def shape(self):
        image_shape = super(SpatialImage3D, self).shape
        spatial_shape = image_shape[:3]
        rest_shape = image_shape[3:]
        if self._affine[0] is not None and self.output_axcodes[0]:
            src_ornt = nib.aff2axcodes(self._affine[0])
            dst_ornt = self.output_axcodes[0]
            src_ornt = nib.orientations.axcodes2ornt(src_ornt)
            dst_ornt = nib.orientations.axcodes2ornt(dst_ornt)
            transf = nib.orientations.ornt_transform(src_ornt, dst_ornt)
            spatial_transf = transf[:,0].astype(np.int).tolist()
            spatial_shape = tuple(spatial_shape[i] for i in spatial_transf)
        if self._original_pixdim[0] and self._output_pixdim[0]:
            zoom_ratio = np.divide(self._original_pixdim[0][:3],
                                   self._output_pixdim[0][:3])
            spatial_shape = tuple(int(round(ii * jj)) for ii, jj in
                                      zip(spatial_shape, zoom_ratio))
        return spatial_shape + rest_shape

    @property
    def output_axcodes(self):
        assert isinstance(self._output_axcodes, tuple)
        return self._output_axcodes

    @output_axcodes.setter
    def output_axcodes(self, output_axcodes):
        self._output_axcodes = output_axcodes
        assert len(self._output_axcodes) == len(self.file_path)

    def load_header(self):
        self._original_pixdim = []
        self._affine = []
        for file_i in self.file_path:
            _obj = misc.load_image(file_i)
            misc.correct_image_if_necessary(_obj)
            self._original_pixdim.append(_obj.header.get_zooms()[:3])
            self._affine.append(_obj.affine)
        self._original_pixdim = tuple(self._original_pixdim)
        self._affine = tuple(self._affine)

    def get_data(self):
        if len(self._file_path) > 1:
            # 3D image from multiple 2d files
            raise NotImplementedError
        # assuming len(self._file_path) == 1
        image_obj = misc.load_image(self.file_path[0])
        image_data = image_obj.get_data()
        image_data = misc.expand_to_5d(image_data)
        if self._affine[0] is not None and self.output_axcodes[0]:
            __orientation = nib.aff2axcodes(self._affine[0])
            image_data = misc.do_reorientation(
                image_data, __orientation, self.output_axcodes[0])

        if self._original_pixdim[0] and self._output_pixdim[0]:
            assert len(self._original_pixdim[0]) == \
                   len(self.output_pixdim[0]), \
                "wrong pixdim format original {} output {}".format(
                    self._original_pixdim[0], self.output_pixdim[0])
            # verbose: warning when interpolate_order>1 for integers
            image_data = misc.do_resampling(image_data,
                                            self._original_pixdim[0],
                                            self.output_pixdim[0],
                                            self.interp_order[0])
        return image_data


class SpatialImage4D(SpatialImage3D):
    def __init__(self,
                 file_path,
                 name,
                 interp_order,
                 output_pixdim,
                 output_axcodes):
        super(SpatialImage4D, self).__init__(file_path=file_path,
                                             name=name,
                                             interp_order=interp_order,
                                             output_pixdim=output_pixdim,
                                             output_axcodes=output_axcodes)

    def get_data(self):
        if len(self._file_path) == 1:
            # 4D image from a single file
            raise NotImplementedError
        # assuming len(self._file_path) > 1
        mod_list = []
        for mod in range(len(self.file_path)):
            __file_path = (self.file_path[mod],)
            __name = (self.name[mod],)
            __interp_order = (self.interp_order[mod],)
            __output_pixdim = (self.output_pixdim[mod],)
            __output_axcodes = (self.output_axcodes[mod],)

            mod_3d = SpatialImage3D(file_path=__file_path,
                                    name=__name,
                                    interp_order=__interp_order,
                                    output_pixdim=__output_pixdim,
                                    output_axcodes=__output_axcodes)
            mod_data_5d = mod_3d.get_data()
            mod_list.append(mod_data_5d)
        try:
            image_data = np.concatenate(mod_list, axis=4)
            # if len(set(self.dtype)) > 1:
            #    tf.logging.warning("cast input images from {} to {}".format(
            #        self.dtype, image_data.dtype))
        except ValueError:
            tf.logging.fatal(
                "multi-modal data shapes not consistent -- trying to "
                "concatenate {}.".format([mod.shape for mod in mod_list]))
            raise
        return image_data


class VectorND(DataFromFile):
    def __init__(self, file_path, name):
        super(VectorND, self).__init__(file_path=file_path,
                                       name=name)

    def get_data(self, resample_to=None, reorient_to=None):
        if resample_to:
            raise NotImplementedError
        if reorient_to:
            raise NotImplementedError
        if len(self._file_path) > 1:
            # 4D image from a single file
            raise NotImplementedError

        image_obj = misc.load_image(self.file_path[0])
        image_data = image_obj.get_data()
        return image_data


class ImageFactory(object):
    INSTANCE_DICT = {2: SpatialImage2D,
                     3: SpatialImage3D,
                     4: SpatialImage4D,
                     5: VectorND}

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
