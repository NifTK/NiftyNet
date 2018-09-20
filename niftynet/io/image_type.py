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
from six import with_metaclass, string_types

import niftynet.io.misc_io as misc
from niftynet.io.image_loader import load_image_obj
from niftynet.io.misc_io import resolve_file_name, dtype_casting
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig


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

    def __init__(self, file_path, name=('loadable_data',), loader=None):
        self._name = None
        self._file_path = None
        self._dtype = None
        self._loader = None

        # assigning using property setters
        self.file_path = file_path
        self.name = name
        self.loader = loader

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
                    load_image_obj(_file, _loader).header.get_data_dtype()
                    for _file, _loader in zip(self.file_path, self.loader))
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
        if isinstance(path_array, string_types):
            path_array = (path_array,)
        home_folder = NiftyNetGlobalConfig().get_niftynet_home_folder()
        try:
            self._file_path = tuple(resolve_file_name(path, ('.', home_folder))
                                    for path in path_array)
        except (TypeError, AssertionError, AttributeError, IOError):
            tf.logging.fatal(
                "unrecognised file path format, should be a valid filename,"
                "or a sequence of filenames %s", path_array)
            raise IOError

    @property
    def loader(self):
        """A tuple of valid image loaders. Always returns a tuple"""
        return self._loader

    @loader.setter
    def loader(self, loader):
        """Makes sure loader is always a tuple of length = #modalities"""
        try:
            if len(self.file_path) == len(loader):
                self._loader = loader
                return
        except TypeError:
            pass
        self._loader = (loader,) * len(self.file_path)

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
        except TypeError:
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
                 output_axcodes,
                 loader):
        DataFromFile.__init__(
            self, file_path=file_path, name=name, loader=loader)
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
    def spatial_rank(self):
        """
        volume [x, y, 1, m, n] will have a spatial rank 2
        volume [x, y, z, m, n] will have a spatial rank 3
           if z > 1

        (resampling/reorientation will not be done when spatial rank is 2).

        """
        return int(np.sum([dim > 1 for dim in self.shape[:3]]))

    @property
    def original_shape(self):
        """
        Shape with multi-modal concatenation, before any resampling.

        :return: a tuple of integers as the original image shape
        """
        return self._original_shape

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
                    load_image_obj(_file, _loader).header['dim'][1:6]
                    for _file, _loader in zip(self.file_path, self.loader))
            except (IOError, KeyError, AttributeError, IndexError):
                tf.logging.fatal(
                    'unknown image shape from header %s', self.file_path)
                raise ValueError
            try:
                non_modality_shapes = \
                    set([tuple(shape[:4].tolist())
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
            self._original_shape = \
                tuple([shape_i if shape_i > 0 else 1
                       for shape_i in self._original_shape])
        return self._original_shape

    def _load_header(self):
        """
        read original header for pixdim and affine info

        :return:
        """
        self._original_pixdim = []
        self._original_affine = []
        for file_i, loader_i in zip(self.file_path, self.loader):
            image_obj = load_image_obj(file_i, loader_i)
            try:
                misc.correct_image_if_necessary(image_obj)
                self._original_pixdim.append(image_obj.header.get_zooms()[:3])
                self._original_affine.append(image_obj.affine)
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
                self._interp_order = \
                    tuple(int(order) for order in interp_order)
                return
        except (TypeError, ValueError):
            pass
        try:
            interp_order = int(interp_order)
            self._interp_order = (int(interp_order),) * len(self.file_path)
        except (TypeError, ValueError):
            tf.logging.fatal(
                "output interp_order should be an integer or "
                "a sequence of integers that matches len(self.file_path)")
            raise ValueError

    @property
    def dtype(self):
        """
        data type property of the input images.

        :return: a tuple of input image data types
            ``len(self.dtype) == len(self.file_path)``
        """
        if not self._dtype:
            self._dtype = super(SpatialImage2D, self).dtype
            self._dtype = tuple(
                dtype_casting(dtype, interp_order)
                for dtype, interp_order in zip(self._dtype, self.interp_order))
        return self._dtype

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
                output_pixdim = \
                    tuple(float(pixdim) for pixdim in output_pixdim)
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
                        self._output_axcodes.append(tuple(output_axcodes[i]))
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

    @classmethod
    def _load_single_file(cls, file_path, loader, dtype=np.float32):
        image_obj = load_image_obj(file_path, loader)
        image_data = image_obj.get_data()  # new API: get_fdata()
        image_data = misc.expand_to_5d(image_data)
        return image_data.astype(dtype)

    def get_data(self):
        if len(self._file_path) > 1:
            image_data = []
            for file_path, loader, dtype in \
                    zip(self._file_path, self.loader, self.dtype):
                data_array = self._load_single_file(file_path, loader, dtype)
                image_data.append(data_array)
            try:
                return np.concatenate(image_data, axis=4)
            except ValueError:
                tf.logging.fatal(
                    "multi-modal data shapes not consistent -- trying to "
                    "concat {}.".format([mod.shape for mod in image_data]))
                raise
        image_data = self._load_single_file(
            self.file_path[0], self.loader[0], self.dtype[0])
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
                 output_axcodes,
                 loader):
        SpatialImage2D.__init__(self,
                                file_path=file_path,
                                name=name,
                                interp_order=interp_order,
                                output_pixdim=output_pixdim,
                                output_axcodes=output_axcodes,
                                loader=loader)
        self._load_header()

    # pylint: disable=no-member
    @SpatialImage2D.output_pixdim.getter
    def output_pixdim(self):
        if self._output_pixdim is None:
            self.output_pixdim = None
        return tuple(self._output_pixdim)

    # pylint: disable=no-member
    @SpatialImage2D.output_axcodes.getter
    def output_axcodes(self):
        if self._output_axcodes is None:
            self.output_axcodes = None
        return tuple(self._output_axcodes)

    @property
    def shape(self):
        image_shape = super(SpatialImage3D, self).shape
        spatial_shape = image_shape[:3]
        rest_shape = image_shape[3:]

        if int(np.sum([dim > 1 for dim in spatial_shape])) < 3:
            # skip resampling and reorientation for spatially 2D
            return image_shape
        pixdim = tuple(self.original_pixdim[0])
        if self.original_axcodes[0] and self.output_axcodes[0]:
            transf, _, _ = misc.compute_orientation(
                self.output_axcodes[0], self.original_axcodes[0])
            spatial_shape = tuple(
                spatial_shape[k] for k in transf[:, 0].astype(np.int))
            if pixdim:
                pixdim = tuple(pixdim[k] for k in transf[:, 0].astype(np.int))

        if pixdim and self.output_pixdim[0]:
            try:
                zoom_ratio = np.divide(pixdim[:3], self.output_pixdim[0][:3])
                spatial_shape = tuple(
                    int(round(ii * jj))
                    for ii, jj in zip(spatial_shape, zoom_ratio))
            except (ValueError, IndexError):
                tf.logging.fatal(
                    'unknown pixdim %s: %s',
                    self.original_pixdim, self.output_pixdim)
                raise ValueError
        return spatial_shape + rest_shape

    def _load_single_file(self, file_path, loader, dtype=np.float32):
        image_data = SpatialImage2D._load_single_file(file_path, loader, dtype)

        if self.spatial_rank < 3:
            return image_data

        pixdim = self.original_pixdim[0]
        if self.original_axcodes[0] and self.output_axcodes[0]:
            image_data = misc.do_reorientation(
                image_data, self.original_axcodes[0], self.output_axcodes[0])
            transf, _, _ = misc.compute_orientation(
                self.output_axcodes[0], self.original_axcodes[0])
            if pixdim:
                pixdim = tuple(pixdim[k] for k in transf[:, 0].astype(np.int))

        if pixdim and self.output_pixdim[0]:
            # verbose: warning when interpolate_order>1 for integers
            image_data = misc.do_resampling(image_data,
                                            pixdim,
                                            self.output_pixdim[0],
                                            self.interp_order[0])
        return image_data


class SpatialImage4D(SpatialImage3D):
    """
    4D image from a set of 3D volumes,
    supports resampling and reorientation.

    The 3D volumes are concatenated in the fifth dim (modality dim)
    """

    def __init__(self,
                 file_path,
                 name,
                 interp_order,
                 output_pixdim,
                 output_axcodes,
                 loader):
        SpatialImage3D.__init__(self,
                                file_path=file_path,
                                name=name,
                                interp_order=interp_order,
                                output_pixdim=output_pixdim,
                                output_axcodes=output_axcodes,
                                loader=loader)

    @property
    def spatial_rank(self):
        """
        Inferring spatial rank from array shape.

        In the case of concatenating ``M`` volumes of ``[x, y, 1]``
        the outcome ``[x, y, 1, 1, M]`` will have a spatial rank 2
        (resampling/reorientation will not be done in this case).

        :return: an integer
        """
        return int(np.sum([dim > 1 for dim in self.shape[:3]]))

    def get_data(self):
        if len(self.file_path) == 1:
            # 4D image from a single file ()
            return SpatialImage3D._load_single_file(
                self, self.file_path[0], self.loader[0])
        # assuming len(self._file_path) > 1
        mod_list = []
        for mod in range(len(self.file_path)):
            mod_3d = SpatialImage3D(file_path=(self.file_path[mod],),
                                    name=(self.name[mod],),
                                    interp_order=(self.interp_order[mod],),
                                    output_pixdim=(self.output_pixdim[mod],),
                                    output_axcodes=(self.output_axcodes[mod],),
                                    loader=(self.loader[mod],))
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


class SpatialImage5D(SpatialImage4D):
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
                 output_axcodes,
                 loader):
        SpatialImage4D.__init__(self,
                                file_path=file_path,
                                name=name,
                                interp_order=interp_order,
                                output_pixdim=output_pixdim,
                                output_axcodes=output_axcodes,
                                loader=loader)


class ImageFactory(object):
    """
    Create image instance according to number of dimensions
    specified in image headers.
    """
    INSTANCE_DICT = {
        2: SpatialImage2D,
        3: SpatialImage3D,
        4: SpatialImage4D,
        5: SpatialImage5D,
        6: SpatialImage5D}

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

        ndims = 0
        image_type = None
        home_folder = NiftyNetGlobalConfig().get_niftynet_home_folder()
        try:
            file_path = resolve_file_name(file_path, ('.', home_folder))
            if os.path.isfile(file_path):
                loader = kwargs.get('loader', None) or None
                ndims = misc.infer_ndims_from_file(file_path, loader)
                image_type = cls.INSTANCE_DICT.get(ndims, None)
        except (TypeError, IOError, AttributeError):
            pass

        if image_type is None:
            try:
                file_path = [
                    resolve_file_name(path, ('.', home_folder))
                    for path in file_path]
                loader = kwargs.get('loader', None) or (None,)
                ndims = misc.infer_ndims_from_file(file_path[0], loader[0])
                ndims = ndims + (1 if len(file_path) > 1 else 0)
                image_type = cls.INSTANCE_DICT.get(ndims, None)
            except (AssertionError, TypeError, IOError, AttributeError):
                tf.logging.fatal('Could not load file: %s', file_path)
                raise IOError
        if image_type is None:
            tf.logging.fatal('Not supported image type from:\n%s', file_path)
            raise NotImplementedError(
                "unrecognised spatial rank {}".format(ndims))
        return image_type(file_path, **kwargs)
