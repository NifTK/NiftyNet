import os

import nibabel as nib
import numpy as np
import misc_io as util
import utilities.constraints_classes as cc
from misc import CacheFunctionOutput

STANDARD_ORIENTATION = [[0, 1], [1, 1], [2, 1]]

class MultiModalFileList(object):
    def __init__(self, multi_mod_filenames):
        # list of multi-modality images filenames
        # each list element is a filename of a single-mod volume
        self.multi_mod_filenames = multi_mod_filenames

    def __call__(self):
        return self.multi_mod_filenames

    @property
    def num_time_point(self):
        if self.multi_mod_filenames is None:
            return 0
        if self.multi_mod_filenames == '':
            return 0
        return len(self.multi_mod_filenames)

    @property
    def num_modality(self):
        if self.multi_mod_filenames is None:
            return 0
        if self.multi_mod_filenames == '':
            return 0
        return len(self.multi_mod_filenames[0])



class Subject(object):
    """
    This class specifies all properties of a subject
    """

    fields = ('input_image_file',
              'target_image_file',
              'weight_map_file',
              'target_note')
    data_types = ('image_filename',
                 'image_filename',
                 'image_filename',
                 'textual_comment')

    def __init__(self, name):
        self.name = name
        self.csv_cell_dict = self._create_empty_csvcell_dict()

        self.is_oriented_to_stand = False
        self.is_isotropic = False

    @classmethod
    def from_csv_row(cls, row):
        new_subject = cls(name=row[0])
        csv_cell_list = [MultiModalFileList(column) if column != '' else None
                         for column in row[1:]]
        new_subject.set_all_columns(*csv_cell_list)
        return new_subject

    def _create_empty_csvcell_dict(self):
        none_tuple = tuple([None] * len(Subject.fields))
        return dict(zip(Subject.fields, none_tuple))

    def set_all_columns(self, *args):
        assert (len(args) == len(Subject.fields))
        for (i, value) in enumerate(args):
            self.set_column(i, value)

    def set_column(self, index, value):
        if value is None:
            return
        assert (isinstance(value, MultiModalFileList))
        self.csv_cell_dict[Subject.fields[index]] = value

    def column(self, index):
        if index > len(Subject.fields) - 1:
            raise ValueError(
                'subject has {} columns, attempting to access index {}'.format(
                    len(Subject.fields), index))
        return self.csv_cell_dict[Subject.fields[index]]

    @CacheFunctionOutput
    def _read_original_affine(self):
        """
        Given the list of files to load, find the original orientation
        and update the corresponding field if not done yet
        """
        img_object = self.__find_first_nibabel_object()
        util.rectify_header_sform_qform(img_object)
        return img_object.affine

    @CacheFunctionOutput
    def _read_original_pixdim(self):
        """
        Given the list of files to load, find the original spatial resolution
        and update the corresponding field if not done yet
        """
        img_object = self.__find_first_nibabel_object()
        return img_object.header.get_zooms()

    def __find_first_nibabel_object(self):
        """
        a helper function find the *first* available image from hard drive
        and return a nibabel image object. This can be used to determine
        image affine/pixel size info.
        This function assumes the header info are the same across all
        volumes for this subject
        :return: nibabel image object
        """
        input_image_files = self.column(0)()
        list_files = [item for sublist in input_image_files for item in sublist]
        for filename in list_files:
            if not filename == '' and os.path.exists(filename):
                path, name, ext = util.split_filename(filename)
                if 'nii' in ext:
                    return nib.load(filename)
        return None

    def __reorient_to_stand(self, data_5d):
        """
        given dictionary data of all modalities,
        this function returns reoriented image data
        """

        if data_5d is None:
            return None
        image_affine = self._read_original_affine()
        ornt_original = nib.orientations.axcodes2ornt(
            nib.aff2axcodes(image_affine))
        return util.do_reorientation(data_5d,
                                     ornt_original,
                                     STANDARD_ORIENTATION)

    def __resample_to_isotropic(self, data_5d, interp_order):
        """
        given image data of all modalities,
        this function returns resampled image data
        """
        if data_5d is None:
            return None
        image_pixdim = self._read_original_pixdim()
        return util.do_resampling(data_5d,
                                  image_pixdim,
                                  [1, 1, 1],
                                  interp_order=interp_order)

    def load_column(self,
                    index,
                    do_reorientation=False,
                    do_resampling=False,
                    interp_order=None):
        # TODO change name to read_image_as_5d
        if Subject.data_types[index] == 'textual_comment':
            return self.column(index)()[0][0]

        elif Subject.data_types[index] == 'image_filename':
            data_5d = util.prepare_5d_data(self.column(index))
            if do_resampling and (interp_order is None):
                print("do resampling, but interpolation order is not "
                      "specified, defaulting to interp_order=3")
                interp_order = 3
                data_5d = self.__resample_to_isotropic(data_5d, interp_order)
            if do_reorientation:
                data_5d = self.__reorient_to_stand(data_5d)
            data_5d = np.nan_to_num(data_5d)
        return {Subject.fields[index]: data_5d}

    def load_columns(self,
                     index_list,
                     do_reorientation=False,
                     do_resampling=False,
                     interp_order=None):
        """
        This function load all images from file_path_list,
        returns all data (with reorientation/resampling if required)
        """

        # set default interp
        if interp_order is None:
            interp_order = [3] * len(index_list)
        if len(interp_order) < len(index_list):
            full_interp_order = [3] * len(index_list)
            full_interp_order[:len(interp_order)] = interp_order
            interp_order = full_interp_order
        output_dict = {}
        for (i, column_ind) in enumerate(index_list):
            column_dict = self.load_column(column_ind,
                                           do_reorientation,
                                           do_resampling,
                                           interp_order[i])
            output_dict[column_dict.keys()[0]] = column_dict.values()[0]
        return output_dict

    def __str__(self):
        out_str = []
        out_str.append('subject: {}'.format(self.name))
        for ind in range(0, len(Subject.fields)):
            csv_field = Subject.fields[ind]
            csv_cell = self.column(ind)
            if csv_cell is None:
                out_str.append('{}: None'.format(csv_field))
            else:
                out_str.append('{}: {}'.format(csv_field, csv_cell()))
        return '\n'.join(out_str)

    def modalities_dict(self):
        num_modality = self.column(0).num_modality
        dict_modalities = {}
        for m in range(0, num_modality):
            name_mod = 'Modality-{}'.format(m)
            dict_modalities[name_mod] = m
        return dict_modalities

    # def _set_data_path_input(self, new_name):
    #    if os.path.exists(new_name):
    #        self.set_column(1, MultiModalFileList([[new_name]]))
    #    else:
    #        warnings.warn("Cannot update new file array as the given file "
    #                      "does not exist")

    # TODO: back to the original volume
    # def _reorient_to_original(self, data):
    #    """
    #    given image data of all modalities in standardised orientation,
    #    this function returns image data in original orientation
    #    """
    #    image_affine = self._read_original_affine()
    #    ornt_original = nib.orientations.axcodes2ornt(
    #            nib.aff2axcodes(image_affine))
    #    data_reoriented = io.do_reorientation(
    #            data, STANDARD_ORIENTATION, ornt_original)
    #    self.is_oriented_to_stand = False
    #    return data_reoriented

    # TODO: support time series
    # def adapt_time_series(self, data):
    #     times = {}
    #     min_times = 1000
    #     max_times = 1
    #     for mod in self.file_path_list.keys():
    #         if data[mod].ndim < 5:
    #             times[mod] = 1
    #             min_times = 1
    #         else:
    #             times[mod] = data[mod].shape[4]
    #             max_times = np.max([times[mod], max_times])
    #             min_times = np.min([times[mod], min_times])
    #     if not min_times == max_times:
    #         warnings.warn("Incompatibility between presented time series")
    #         for mod in self.file_path_list.keys():
    #             if times[mod] < max_times:
    #                 data[mod] = io.adjust_to_maxtime(data[mod], max_times)
    #
    #     return data
