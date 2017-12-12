# -*- coding: utf-8 -*-
"""
This module defines images used by image reader, image properties
are set by user or read from image header.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from abc import ABCMeta, abstractmethod

import nibabel as nib
import numpy as np
import tensorflow as tf
from six import with_metaclass

import niftynet.io.misc_io as misc


class Loadable(with_metaclass(ABCMeta, object)):
    """
    interface of loadable data
    """

    @abstractmethod
    def get_data(self):
        """
        loads a numpy array from the image object
        if the array has less than 5 dimensions
        it extends the array to 5d
        (corresponding to 3 spatial dimensions, temporal dim, modalities)
        ndims > 5 not currently supported
        """
        raise NotImplementedError


class DataFromFile(Loadable):
    """
    Data from file should have a valid file path
    (are files on hard drive) and a name.
    """

    def __init__(self, file_path, name='loadable_data'):
        self._name = None
        self._file_path = None

        # assigning using property setters
        self.file_path = file_path
        self.name = name
        self._dtype = None

    @property
    def dtype(self):
        """
        data type property of the input images.

        :return: a tuple of input image data types
            ``len(self.dtype) == len(self.file_path)``
        """
        if not self._dtype:
            try:
                self._dtype = tuple(
                    misc.load_image(_file).header.get_data_dtype()
                    for _file in self.file_path)
            except (IOError, TypeError, AttributeError):
                tf.logging.warning('could not decide image data type')
                self._dtype = (np.dtype(np.float32),) * len(self.file_path)
        return self._dtype

    @property
    def file_path(self):
        """
        A tuple of valid image filenames, this property always returns
        a tuple, length of the tuple is one for single image,
        length of the tuple is larger than one for single image from
        multiple files.

        :return: a tuple of file paths
        """
        return self._file_path

    @file_path.setter
    def file_path(self, path_array):
        try:
            if os.path.isfile(path_array):
                self._file_path = (os.path.abspath(path_array),)
                return
        except (TypeError, AttributeError):
            pass
        try:
            assert all([os.path.isfile(file_name) for file_name in path_array])
            self._file_path = \
                tuple(os.path.abspath(file_name) for file_name in path_array)
            return
        except (TypeError, AssertionError, AttributeError):
            tf.logging.fatal(
                "unrecognised file path format, should be a valid filename,"
                "or a sequence of filenames %s", path_array)
            raise IOError

    @property
    def name(self):
        """
        A tuple of image names, this property always returns
        a tuple, length of the tuple is one for single image,
        length of the tuple is larger than one for single image from
        multiple files.

        :return: a tuple of image name tags
        """
        return self._name

    @name.setter
    def name(self, name_array):
        try:
            if len(self.file_path) == len(name_array):
                self._name = name_array
                return
        except (TypeError, AssertionError):
            pass
        self._name = (name_array,)

    def get_data(self):
        raise NotImplementedError


class SpatialImage2D(DataFromFile):
    """
    2D images, axcodes specifications are ignored when
    loading. (Resampling to new pixdims is currently not supported).
    """

    def __init__(self,
                 file_path,
                 name,
                 interp_order,
                 output_pixdim,
                 output_axcodes):
        DataFromFile.__init__(self, file_path=file_path, name=name)
        self._original_affine = None
        self._original_pixdim = None
        self._original_shape = None
        self._interp_order = None
        self._output_pixdim = None
        self._output_axcodes = None

        # assigning with property setters
        self.interp_order = interp_order
        self.output_pixdim = output_pixdim
        self.output_axcodes = output_axcodes

        self._load_header()

    @property
    def shape(self):
        """
        This function read image shape info from the headers
        The lengths in the fifth dim of multiple images are summed
        as a multi-mod representation.
        The fourth dim corresponding to different time sequences
        is ignored.

        :return: a tuple of integers as image shape
        """
        if self._original_shape is None:
            try:
                self._original_shape = tuple(
                    misc.load_image(_file).header['dim'][1:6]
                    for _file in self.file_path)
            except (IOError, KeyError, AttributeError, IndexError):
                tf.logging.fatal(
                    'unknown image shape from header %s', self.file_path)
                raise ValueError
            try:
                non_modality_shapes = set(
                    [tuple(shape[:4].tolist())
                     for shape in self._original_shape])
                assert len(non_modality_shapes) == 1
            except (TypeError, IndexError, AssertionError):
                tf.logging.fatal("could not combining multimodal images: "
                                 "shapes not consistent %s -- %s",
                                 self.file_path, self._original_shape)
                raise ValueError
            n_modalities = \
                np.sum([int(shape[4]) for shape in self._original_shape])
            self._original_shape = non_modality_shapes.pop() + (n_modalities,)
        return self._original_shape

    def _load_header(self):
        """
        read original header for pixdim and affine info

        :return:
        """
        self._original_pixdim = []
        self._original_affine = []
        for file_i in self.file_path:
            _obj = misc.load_image(file_i)
            try:
                misc.correct_image_if_necessary(_obj)
                self._original_pixdim.append(_obj.header.get_zooms()[:3])
                self._original_affine.append(_obj.affine)
            except (TypeError, IndexError, AttributeError):
                tf.logging.fatal('could not read header from %s', file_i)
                raise ValueError
                # self._original_pixdim = tuple(self._original_pixdim)
                # self._original_affine = tuple(self._original_affine)

    @property
    def original_pixdim(self):
        """
        pixdim info from the image header.

        :return: a tuple of pixdims, with each element as pixdims
            of an image file
        """
        try:
            assert self._original_pixdim[0] is not None
        except (IndexError, AssertionError):
            self._load_header()
        return self._original_pixdim

    @property
    def original_affine(self):
        """
        affine info from the image header.

        :return: a tuple of affine, with each element as an affine
            matrix of an image file
        """
        try:
            assert self._original_affine[0] is not None
        except (IndexError, AssertionError):
            self._load_header()
        return self._original_affine

    @property
    def original_axcodes(self):
        """
        axcodes info from the image header
        more info: http://nipy.org/nibabel/image_orientation.html

        :return: a tuple of axcodes, with each element as axcodes
            of an image file
        """
        try:
            return tuple(nib.aff2axcodes(affine)
                         for affine in self.original_affine)
        except IndexError:
            tf.logging.fatal('unknown affine in header %s: %s',
                             self.file_path, self.original_affine)
            raise

    @property
    def interp_order(self):
        """
        interpolation order specified by user.

        :return: a tuple of integers, with each element as an
            interpolation order of an image file
        """
        return self._interp_order

    @interp_order.setter
    def interp_order(self, interp_order):
        try:
            if len(interp_order) == len(self.file_path):
                self._interp_order = tuple(int(order) for order in interp_order)
                return
        except (TypeError, ValueError):
            pass
        try:
            interp_order = int(interp_order)
            self._interp_order = (int(interp_order),) * len(self.file_path)
        except (TypeError, ValueError):
            tf.logging.fatal(
                "output interp_order should be an integer or"
                "a sequence of integers that matches len(self.file_path)")
            raise ValueError

    @property
    def output_pixdim(self):
        """
        output pixdim info specified by user
        set to None for using the original pixdim in image header
        otherwise get_data() transforms image array according to this value.

        :return: a tuple of pixdims, with each element as pixdims
            of an image file
        """
        tf.logging.warning("resampling 2D images not implemented")
        return (None,) * len(self.file_path)

    @output_pixdim.setter
    def output_pixdim(self, output_pixdim):
        try:
            if len(output_pixdim) == len(self.file_path):
                self._output_pixdim = []
                for i, _ in enumerate(self.file_path):
                    if output_pixdim[i] is None:
                        self._output_pixdim.append(None)
                    else:
                        self._output_pixdim.append(
                            tuple(float(pixdim) for pixdim in output_pixdim[i]))
                # self._output_pixdim = tuple(self._output_pixdim)
                return
        except (TypeError, ValueError):
            pass
        try:
            if output_pixdim is not None:
                output_pixdim = tuple(float(pixdim) for pixdim in output_pixdim)
            self._output_pixdim = (output_pixdim,) * len(self.file_path)
        except (TypeError, ValueError):
            tf.logging.fatal(
                'could not set output pixdim '
                '%s for %s', output_pixdim, self.file_path)
            raise

    @property
    def output_axcodes(self):
        """
        output axcodes info specified by user
        set to None for using the original axcodes in image header,
        otherwise get_data() change axes of the image array
        according to this value.

        :return: a tuple of pixdims, with each element as pixdims
            of an image file
        """
        tf.logging.warning("reorienting 2D images not implemented")
        return (None,) * len(self.file_path)

    @output_axcodes.setter
    def output_axcodes(self, output_axcodes):
        try:
            if len(output_axcodes) == len(self.file_path):
                self._output_axcodes = []
                for i, _ in enumerate(self.file_path):
                    if output_axcodes[i] is None:
                        self._output_axcodes.append(None)
                    else:
                        self._output_axcodes.append(
                            tuple(output_axcodes[i]))
                # self._output_axcodes = tuple(self._output_axcodes)
                return
        except (TypeError, ValueError):
            pass
        try:
            if output_axcodes is None:
                output_axcodes = (None,)
            else:
                output_axcodes = (output_axcodes,)
            self._output_axcodes = output_axcodes * len(self.file_path)
        except (TypeError, ValueError):
            tf.logging.fatal(
                'could not set output pixdim '
                '%s for %s', output_axcodes, self.file_path)
            raise

    def get_data(self):
        if len(self._file_path) > 1:
            # 2D image from multiple files
            raise NotImplementedError
        image_obj = misc.load_image(self.file_path[0])
        image_data = image_obj.get_data()
        image_data = misc.expand_to_5d(image_data)
        return image_data


class SpatialImage3D(SpatialImage2D):
    """
    3D image from a single, supports resampling and reorientation
    (3D image from a set of 2D slices is currently not supported).
    """

    def __init__(self,
                 file_path,
                 name,
                 interp_order,
                 output_pixdim,
                 output_axcodes):
        SpatialImage2D.__init__(self,
                                file_path=file_path,
                                name=name,
                                interp_order=interp_order,
                                output_pixdim=output_pixdim,
                                output_axcodes=output_axcodes)
        self._load_header()

    # pylint: disable=no-member
    @SpatialImage2D.output_pixdim.getter
    def output_pixdim(self):
        if self._output_pixdim is None:
            self.output_pixdim = None
        return self._output_pixdim

    # pylint: disable=no-member
    @SpatialImage2D.output_axcodes.getter
    def output_axcodes(self):
        if self._output_axcodes is None:
            self.output_axcodes = None
        return self._output_axcodes

    @property
    def shape(self):
        image_shape = super(SpatialImage3D, self).shape
        spatial_shape = image_shape[:3]
        rest_shape = image_shape[3:]
        if self.original_affine[0] is not None and self.output_axcodes[0]:
            src_ornt = nib.orientations.axcodes2ornt(self.original_axcodes[0])
            dst_ornt = nib.orientations.axcodes2ornt(self.output_axcodes[0])
            if np.any(np.isnan(dst_ornt)) or np.any(np.isnan(src_ornt)):
                tf.logging.fatal(
                    'unknown output axcodes %s for %s',
                    self.output_axcodes, self.original_axcodes)
                raise ValueError
            transf = nib.orientations.ornt_transform(src_ornt, dst_ornt)
            spatial_transf = transf[:, 0].astype(np.int).tolist()
            new_shape = [0, 0, 0]
            for i, k in enumerate(spatial_transf):
                new_shape[k] = spatial_shape[i]
            spatial_shape = tuple(new_shape)
        if self.original_pixdim[0] and self.output_pixdim[0]:
            try:
                zoom_ratio = np.divide(self.original_pixdim[0][:3],
                                       self.output_pixdim[0][:3])
                spatial_shape = tuple(int(round(ii * jj)) for ii, jj in
                                      zip(spatial_shape, zoom_ratio))
            except (ValueError, IndexError):
                tf.logging.fatal(
                    'unknown pixdim %s: %s',
                    self.original_pixdim, self.output_pixdim)
                raise ValueError
        return spatial_shape + rest_shape

    def get_data(self):
        if len(self._file_path) > 1:
            # 3D image from multiple 2d files
            raise NotImplementedError
        # assuming len(self._file_path) == 1
        image_obj = misc.load_image(self.file_path[0])
        image_data = image_obj.get_data()
        image_data = misc.expand_to_5d(image_data)
        if self.original_axcodes[0] and self.output_axcodes[0]:
            image_data = misc.do_reorientation(
                image_data, self.original_axcodes[0], self.output_axcodes[0])

        if self.original_pixdim[0] and self.output_pixdim[0]:
            # verbose: warning when interpolate_order>1 for integers
            image_data = misc.do_resampling(image_data,
                                            self.original_pixdim[0],
                                            self.output_pixdim[0],
                                            self.interp_order[0])
        return image_data


class SpatialImage4D(SpatialImage3D):
    """
    4D image from a set of 3D volumes,
    supports resampling and reorientation.

    The 3D volumes are concatenated in the fifth dim (modality dim)
    (4D image from a single file is currently not supported)
    """

    def __init__(self,
                 file_path,
                 name,
                 interp_order,
                 output_pixdim,
                 output_axcodes):
        SpatialImage3D.__init__(self,
                                file_path=file_path,
                                name=name,
                                interp_order=interp_order,
                                output_pixdim=output_pixdim,
                                output_axcodes=output_axcodes)

    def get_data(self):
        if len(self.file_path) == 1:
            # 4D image from a single file
            raise NotImplementedError(
                "loading 4D image (time sequence) is not supported")
        # assuming len(self._file_path) > 1
        mod_list = []
        for mod in range(len(self.file_path)):
            mod_3d = SpatialImage3D(file_path=(self.file_path[mod],),
                                    name=(self.name[mod],),
                                    interp_order=(self.interp_order[mod],),
                                    output_pixdim=(self.output_pixdim[mod],),
                                    output_axcodes=(self.output_axcodes[mod],))
            mod_data_5d = mod_3d.get_data()
            mod_list.append(mod_data_5d)
        try:
            image_data = np.concatenate(mod_list, axis=4)
        except ValueError:
            tf.logging.fatal(
                "multi-modal data shapes not consistent -- trying to "
                "concatenate {}.".format([mod.shape for mod in mod_list]))
            raise
        return image_data


class SpatialImage5D(SpatialImage3D):
    """
    5D image from a single file,
    resampling and reorientation are implemented as
    operations on each 3D slice individually.

    (5D image from a set of 4D files is currently not supported)
    """

    def __init__(self,
                 file_path,
                 name,
                 interp_order,
                 output_pixdim,
                 output_axcodes):
        SpatialImage3D.__init__(self,
                                file_path=file_path,
                                name=name,
                                interp_order=interp_order,
                                output_pixdim=output_pixdim,
                                output_axcodes=output_axcodes)

    def _load_single_5d(self, idx=0):
        if len(self._file_path) > 1:
            # 3D image from multiple 2d files
            raise NotImplementedError
        # assuming len(self._file_path) == 1
        image_obj = misc.load_image(self.file_path[idx])
        image_data = image_obj.get_data()
        image_data = misc.expand_to_5d(image_data)
        assert image_data.shape[3] == 1, "time sequences not supported"
        if self.original_axcodes[idx] and self.output_axcodes[idx]:
            output_image = []
            for t_pt in range(image_data.shape[3]):
                mod_list = []
                for mod in range(image_data.shape[4]):
                    spatial_slice = image_data[..., t_pt:t_pt + 1, mod:mod + 1]
                    spatial_slice = misc.do_reorientation(
                        spatial_slice,
                        self.original_axcodes[idx],
                        self.output_axcodes[idx])
                    mod_list.append(spatial_slice)
                output_image.append(np.concatenate(mod_list, axis=4))
            image_data = np.concatenate(output_image, axis=3)

        if self.original_pixdim[idx] and self.output_pixdim[idx]:
            assert len(self._original_pixdim[idx]) == \
                   len(self.output_pixdim[idx]), \
                   "wrong pixdim format original {} output {}".format(
                       self._original_pixdim[idx], self.output_pixdim[idx])
            # verbose: warning when interpolate_order>1 for integers
            output_image = []
            for t_pt in range(image_data.shape[3]):
                mod_list = []
                for mod in range(image_data.shape[4]):
                    spatial_slice = image_data[..., t_pt:t_pt + 1, mod:mod + 1]
                    spatial_slice = misc.do_resampling(
                        spatial_slice,
                        self.original_pixdim[idx],
                        self.output_pixdim[idx],
                        self.interp_order[idx])
                    mod_list.append(spatial_slice)
                output_image.append(np.concatenate(mod_list, axis=4))
            image_data = np.concatenate(output_image, axis=3)
        return image_data

    def get_data(self):
        if len(self._file_path) == 1:
            return self._load_single_5d()
        else:
            raise NotImplementedError('concatenating 5D images not supported.')
            #     image_data = []
            #     for idx in range(len(self._file_path)):
            #         image_data.append(self._load_single_5d(idx))
            #     image_data = np.concatenate(image_data, axis=4)
            # return image_data


class ImageFactory(object):
    """
    Create image instance according to number of dimensions
    specified in image headers.
    """
    INSTANCE_DICT = {2: SpatialImage2D,
                     3: SpatialImage3D,
                     4: SpatialImage4D,
                     5: SpatialImage5D}

    @classmethod
    def create_instance(cls, file_path, **kwargs):
        """
        Read image headers and create image instance.

        :param file_path: a file path or a sequence of file paths
        :param kwargs: output properties for transforming the image data
            array into a desired format
        :return: an image instance
        """
        if file_path is None:
            tf.logging.fatal('No file_path provided, '
                             'please check input sources in config file')
            raise ValueError
        image_type = None
        try:
            if os.path.isfile(file_path):
                ndims = misc.infer_ndims_from_file(file_path)
                image_type = cls.INSTANCE_DICT.get(ndims, None)
        except TypeError:
            pass
        if image_type is None:
            try:
                assert all([os.path.isfile(path) for path in file_path])
                ndims = misc.infer_ndims_from_file(file_path[0])
                ndims = ndims + (1 if len(file_path) > 1 else 0)
                image_type = cls.INSTANCE_DICT.get(ndims, None)
            except AssertionError:
                tf.logging.fatal('Could not load file: %s', file_path)
                raise IOError
        if image_type is None:
            tf.logging.fatal('Not supported image type: %s', file_path)
            raise NotImplementedError
        return image_type(file_path, **kwargs)
