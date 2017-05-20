import nibabel as nib
import misc_io as util
from misc import CacheFunctionOutput
import numpy as np

STANDARD_ORIENTATION = [[0, 1], [1, 1], [2, 1]]

class Subject(object):
    """
    This class specifies all properties of a subject
    """
    def __init__(self,
                 name,
                 file_path_dict,
                 list_nn,
                 allow_multimod_single_file=False,
                 allow_timeseries=False):

        self.name = name
        self.file_path_dict = file_path_dict
        self.list_nn = list_nn

        self.is_oriented_to_stand = False
        self.is_isotropic = False

        self.allow_multimod_single_file = allow_multimod_single_file
        self.allow_timeseries = allow_timeseries

    @CacheFunctionOutput
    def _read_original_affine(self):
        """
        Given the list of files to load, find the original orientation
        and update the corresponding field if not done yet
        """
        modality_list = self.file_path_dict.keys()
        filename_first = self.file_path_dict[modality_list[0]]
        img_original = nib.load(filename_first)
        util.rectify_header_sform_qform(img_original)
        #print img_original.affine
        return img_original.affine

    @CacheFunctionOutput
    def _read_original_pixdim(self):
        """
        Given the list of files to load, find the original spatial resolution
        and update the corresponding field if not done yet
        """
        modality_list = self.file_path_dict.keys()
        filename_first = self.file_path_dict[modality_list[0]]
        img_original = nib.load(filename_first)
        #print img_original.header.get_zooms()
        return img_original.header.get_zooms()

    def _set_data_path(self, file_path, modality):
        # TODO: check file exists
        self.file_path_dict[modality] = file_path

    def _reorient_to_stand(self, data):
        """
        given dictionary data of all modalities,
        this function returns reoriented image data
        """
        image_affine = self._read_original_affine()
        ornt_original = nib.orientations.axcodes2ornt(
                nib.aff2axcodes(image_affine))
        data_reoriented = util.do_reorientation(
                data, ornt_original, STANDARD_ORIENTATION)
        self.is_oriented_to_stand = True
        return data_reoriented

    def _resample_to_isotropic(self, data):
        """
        given image data of all modalities,
        this function returns resampled image data
        """
        image_pixdim = self._read_original_pixdim()
        isotropic_pixdim = [1, 1, 1]
        for mod in self.file_path_dict.keys():
            interp_order = 0 if mod in self.list_nn else 3
            new_data = util.do_resampling(data[mod],
                                        image_pixdim,
                                        isotropic_pixdim,
                                        interp_order=interp_order)
            data[mod] = new_data
        self.is_isotropic = True
        return data

    def read_all_modalities(self, do_reorient=False, do_resample=False):
        """
        This function load all images from file_path_dict,
        returns all data (with reorientation/resampling if required)
        """
        data = {}
        for mod in self.file_path_dict.keys():
            data[mod] = util.load_volume(self.file_path_dict[mod],
                                       self.allow_multimod_single_file,
                                       self.allow_timeseries)
            if data[mod] is None:
                self.file_path_dict.remove(mod)
                del data[mod]
        if self.allow_timeseries:
            # self.adapt_time_series(data) Not in use yet
            raise NotImplementedError
        if do_resample:
            data = self._resample_to_isotropic(data)
        if do_reorient:
            data = self._reorient_to_stand(data)
        data = self._expand_to_4d(data)
        return data

    # TODO: merge this function to misc_io.py: load_volume()
    def _expand_to_4d(self, data):
        for mod in self.file_path_dict.keys():
            if data[mod].ndim == 3:
                data[mod] = np.expand_dims(data[mod], axis=3)
        return data

    def __str__(self):
        out_str = 'subject: {}.'.format(self.name)
        out_str += ' file_path: {}.'.format(self.file_path_dict)
        out_str += ' do reorientation: {}.'.format(self.is_oriented_to_stand)
        out_str += ' do resampling: {}.'.format(self.is_isotropic)
        return out_str


    # TODO: back to the original volume
    #def _reorient_to_original(self, data):
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


    # def adapt_time_series(self, data):
    #     times = {}
    #     min_times = 1000
    #     max_times = 1
    #     for mod in self.file_path_dict.keys():
    #         if data[mod].ndim < 5:
    #             times[mod] = 1
    #             min_times = 1
    #         else:
    #             times[mod] = data[mod].shape[4]
    #             max_times = np.max([times[mod], max_times])
    #             min_times = np.min([times[mod], min_times])
    #     if not min_times == max_times:
    #         warnings.warn("Incompatibility between presented time series")
    #         for mod in self.file_path_dict.keys():
    #             if times[mod] < max_times:
    #                 data[mod] = io.adjust_to_maxtime(data[mod], max_times)
    #
    #     return data
