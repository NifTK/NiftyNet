import nibabel as nib

import misc_io as util
import utilities.constraints_classes as cc
from misc import CacheFunctionOutput

STANDARD_ORIENTATION = [[0, 1], [1, 1], [2, 1]]


class Subject(object):
    """
    This class specifies all properties of a subject
    """

    def __init__(self,
                 name,
                 file_path_list,
                 interp_order):

        self.name = name
        self.file_path_list = file_path_list
        self.interp_order = interp_order

        self.is_oriented_to_stand = False
        self.is_isotropic = False

    @CacheFunctionOutput
    def _read_original_affine(self):
        """
        Given the list of files to load, find the original orientation
        and update the corresponding field if not done yet
        """
        filename_first = self.file_path_list.input.filename_ref
        img_original = nib.load(filename_first)
        util.rectify_header_sform_qform(img_original)
        # print img_original.affine
        return img_original.affine

    @CacheFunctionOutput
    def _read_original_pixdim(self):
        """
        Given the list of files to load, find the original spatial resolution
        and update the corresponding field if not done yet
        """
        filename_first = self.file_path_list.input.filename_ref
        img_original = nib.load(filename_first)
        # print img_original.header.get_zooms()
        return img_original.header.get_zooms()

    def _set_data_path(self, new_name):
        # TODO: check file exists
        self.file_path_list.input.filename_ref = new_name

    def _reorient_to_stand(self, data):
        """
        given dictionary data of all modalities,
        this function returns reoriented image data
        """
        image_affine = self._read_original_affine()
        ornt_original = nib.orientations.axcodes2ornt(
            nib.aff2axcodes(image_affine))
        if not data.input is None:
            data.input = util.do_reorientation(
                data.input, ornt_original, STANDARD_ORIENTATION)
        if not data.output is None:
            data.output = util.do_reorientation(
                data.output, ornt_original, STANDARD_ORIENTATION)
        if not data.weight is None:
            data.weight = util.do_reorientation(
                data.weight, ornt_original, STANDARD_ORIENTATION)
        self.is_oriented_to_stand = True
        return data

    def _resample_to_isotropic(self, data):
        """
        given image data of all modalities,
        this function returns resampled image data
        """
        image_pixdim = self._read_original_pixdim()
        if not data.input is None:
            data.input = util.do_resampling(
                    data.input,
                    image_pixdim,
                    [1, 1, 1],
                    interp_order=self.interp_order.input)

        if not data.output is None:
            data.output = util.do_resampling(
                    data.output,
                    image_pixdim,
                    [1, 1, 1],
                    interp_order=self.interp_order.output)

        if not data.weight is None:
            data.weight = util.do_resampling(
                    data.weight,
                    image_pixdim,
                    [1, 1, 1],
                    interp_order=self.interp_order.weight)
        self.is_isotropic = True
        return data

    def read_all_modalities(self, do_reorient=False, do_resample=False):
        """
        This function load all images from file_path_list,
        returns all data (with reorientation/resampling if required)
        """
        data_input = util.prepare_5d_data(self.file_path_list.input)
        data_output = util.prepare_5d_data(self.file_path_list.output)
        data_weight = util.prepare_5d_data(self.file_path_list.weight)
        data = cc.InputList(data_input, data_output, data_weight, None, None)
        if do_resample:
            data = self._resample_to_isotropic(data)
        if do_reorient:
            data = self._reorient_to_stand(data)
        return data

    def __str__(self):
        out_str = 'subject: {}.'.format(self.name)
        out_str += ' file_path: {}.'.format(self.file_path_list)
        out_str += ' do reorientation: {}.'.format(self.is_oriented_to_stand)
        out_str += ' do resampling: {}.'.format(self.is_isotropic)
        return out_str


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
