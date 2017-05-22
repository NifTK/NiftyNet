import csv
import warnings
from random import shuffle

import numpy as np

import misc_io as io
import nn.histogram_standardisation as hs
import utilities.constraints_classes as cc
import utilities.misc_csv as misc_csv
from nn.preprocess import HistNormaliser_bis
from utilities.subject import Subject


class VolumePreprocessor(object):
    """
    This class manages the loading step, i.e., return subject's data
    by searching user provided path and modality constraints.
    The volumes are resampled/reoriented if required.

    This class maintains a list of subjects, where each element of the list
    is a Patient object.
    """

    def __init__(self,
                 dict_normalisation,
                 dict_masking,
                 csv_file=None,
                 csv_list=None,
                 number_list=None,
                 flags=cc.Flags(),
                 loss=['dice']):

        self.dict_normalisation = dict_normalisation
        self.dict_masking = dict_masking
        self.csv_list = csv_list
        self.csv_file = csv_file
        self.number_list = number_list
        self.flags = flags
        self.loss = loss

        self.standardisor = HistNormaliser_bis(
            self.dict_normalisation.hist_ref_file,
            self.dict_normalisation.path_to_train,
            self.dict_normalisation.norm_type, self.dict_normalisation.cutoff,
            self.dict_masking.mask_type, '')

        self.subject_list = self._search_for_eligible_subjects()
        self.current_id = 0

    def create_list_subject_from_csv(self):
        list_subjects = []
        interp_order = self.guess_interp_from_loss()
        interp_order_fields = cc.InputList([3], interp_order, [3], None, None)
        with open(self.csv_file, "rb") as infile:
            reader = csv.reader(infile)
            for row in reader:
                input = cc.InputFiles(row[1], [[row[1]]])
                output = cc.InputFiles(row[2], [[row[2]]])
                weight = cc.InputFiles(row[3], [[row[3]]])
                data_list = cc.InputList(input, output, weight, row[4], row[5])
                new_subject = Subject(row[0], data_list,
                                      interp_order=interp_order_fields)
                list_subjects.append(new_subject)
        return list_subjects

    def guess_interp_from_loss(self):
        categorical = ['cross_entropy', 'dice']
        interp_order = []
        for l in self.loss:
            order = 0 if l in categorical else 3
            interp_order.append(order)
        return interp_order

    def create_array_subjects_csv_list(self):
        subjects_input, files_input = misc_csv.create_array_files_from_csv(
            self.csv_list.input, self.number_list.input,
            self.flags.flag_allow_missing)
        subjects_output, files_output = misc_csv.create_array_files_from_csv(
            self.csv_list.output, self.number_list.output,
            self.flags.flag_allow_missing)
        subjects_weight, files_weight = misc_csv.create_array_files_from_csv(
            self.csv_list.weight, self.number_list.weight,
            self.flags.flag_allow_missing)
        subjects_input_txt, files_input_txt = \
            misc_csv.create_array_files_from_csv(self.csv_list.input_txt,
                                                 self.number_list.input_txt,
                                                 self.flags.flag_allow_missing)
        subjects_output_txt, files_output_txt = \
            misc_csv.create_array_files_from_csv(self.csv_list.output_txt,
                                                 self.number_list.output_txt,
                                                 self.flags.flag_allow_missing)
        subjects_input = [[subject] for subject in subjects_input]
        subjects_output = [[subject] for subject in subjects_output]
        subjects_weight = [[subject] for subject in subjects_weight] if \
            subjects_weight is not None else None
        subjects_input_txt = [[subject] for subject in subjects_input_txt] if \
            subjects_input_txt is not None else None
        subjects_output_txt = [[subject] for subject in subjects_output_txt] if \
            subjects_output_txt is not None else None
        name_list = cc.InputList(subjects_input, subjects_output,
                                 subjects_weight, subjects_input_txt,
                                 subjects_output_txt)
        file_list = cc.InputList(files_input, files_output, files_weight,
                                 files_input_txt, files_output_txt)
        list_combined = misc_csv.combine_list_constraint(name_list, file_list)
        return list_combined

    def create_list_subject_from_list(self):
        interp_order = self.guess_interp_from_loss()
        interp_order_fields = cc.InputList([3], interp_order, [3], None, None)
        list_combined = self.create_array_subjects_csv_list()
        subjects = []
        for s in list_combined:
            input_files = cc.InputFiles(s[1][0][0], s[1])
            output_files = cc.InputFiles(s[2][0][0], s[2])
            weight_files = cc.InputFiles(s[3][0][0], s[3]) if s[3] is not '' else None
            input_txt_files = cc.InputFiles(s[4][0][0], s[4]) if s[4] is not '' else None
            output_txt_files = cc.InputFiles(s[5][0][0], s[5]) if s[5] is not '' else None
            file_path_list = cc.InputList(input_files,
                                          output_files,
                                          weight_files,
                                          input_txt_files,
                                          output_txt_files)

            new_subject = Subject(s[0], file_path_list, interp_order_fields)
            subjects.append(new_subject)
        return subjects

    # Provide the final list of eligible subjects
    def _search_for_eligible_subjects(self):
        modalities = self.dict_normalisation.list_modalities
        if self.csv_file is None and self.csv_list is None:
            raise ValueError("There is not input to build the subjects list!!!")
        if self.csv_file is None:
            subjects = self.create_list_subject_from_list()
        else:
            subjects = self.create_list_subject_from_csv()

        if self.flags.flag_standardise and len(hs.check_modalities_to_train(
                self.dict_normalisation.hist_ref_file,
                modalities.keys())) > 0:
            mod_to_train = hs.check_modalities_to_train(
                self.dict_normalisation.hist_ref_file,
                modalities.keys())
            modalities_to_train = {}
            for mod in mod_to_train:
                modalities_to_train[mod] = modalities[mod]
            warnings.warn("The histogram has to be retrained...")
            array_files = misc_csv.create_array_files(csv_file=
                                                      self.csv_file,
                                                      csv_list=
                                                      self.csv_list)

            new_mapping = self.standardisor \
                .training_normalisation_from_array_files(
                array_files, modalities_to_train)
            self.standardisor.complete_and_transform_model_file(
                new_mapping, mod_to_train)
        if self.flags.flag_standardise and self.flags.flag_save_norm:
            for s in subjects:
                self.normalise_subject_data_and_save(s)

        return subjects

    def whiten_subject_data_array(self, data_array, modalities_indices=None):
        if modalities_indices is None:
            modalities_indices = range(0, data_array.shape[3])
        list_mod_whiten = [m for m in modalities_indices if
                           m < data_array.shape[3]]
        mask_array = self.make_mask_array(data_array)
        for m in list_mod_whiten:
            for t in range(0, data_array.shape[4]):
                data_array[..., m, t] = \
                    self.standardisor.whitening_transformation(
                        data_array[..., m, t], mask_array[..., m, t])
        return data_array

    def whiten_subject_data(self, data_dict, modalities):
        mask_array = self.make_mask_array(data_dict.input)
        for m in modalities:
            for t in range(0, len(data_dict[m])):
                data_dict.input[..., m, t] = self.standardisor. \
                    whitening_transformation(
                    data_dict.input[..., m, t], mask_array[..., m, t])
        return data_dict

    def normalise_subject_data_and_save(self, subject):
        if self.flags.flag_standardise:
            data_dict = subject.read_all_modalities(self.flags.flag_reorient,
                                                    self.flags.flag_resample)
            data_dict.input = np.nan_to_num(data_dict.input)
            mask_array = self.make_mask_array(data_dict.input)
            data_dict.input = self.standardisor.normalise_data_array(
                data_dict.input, mask_array)
            name_norm_save = io.create_new_filename(
                subject.name + '.nii.gz',
                new_path=self.dict_normalisation.path_to_save,
                new_prefix='Norm')
            data_nifti_format = np.swapaxes(data_dict.input, 4, 3)  # Put
            # back the array with the nifti conventions.
            io.save_img(data_nifti_format, subject.name, [], name_norm_save,
                        filename_ref=subject.file_path_list.input.filename_ref,
                        flag_orientation=self.flags.flag_reorient,
                        flag_isotropic=self.flags.flag_resample)
            subject._set_data_path(name_norm_save)

    def make_mask_array(self, data_array):
        data_array = io.expand_for_5d(data_array)
        max_time = data_array.shape[4]
        list_indices_mod = [m for m in range(0, data_array.shape[3]) if
                            np.count_nonzero(data_array[..., m, :]) > 0]
        mod_masking = list_indices_mod
        mask_array = np.zeros_like(data_array)
        for mod in mod_masking:
            for t in range(0, max_time):
                new_mask = hs.create_mask_img_3d(
                    data_array[..., mod, t], self.dict_masking.mask_type)
                new_mask = io.expand_for_5d(new_mask)
                mask_array[..., mod:mod + 1, t:t + 1] = new_mask
        if self.dict_masking.multimod_type == 'or':
            for t in range(0, max_time):
                new_mask = np.zeros([data_array.shape[0:3]])
                for mod in mod_masking:
                    if np.count_nonzero(data_array[..., mod, t]) > 0:
                        new_mask = new_mask + mask_array[..., mod, t]
                new_mask[new_mask > 0.5] = 1
                mask_array[..., t] = io.expand_for_5d(np.tile(np.expand_dims(
                    new_mask, axis=3), [1, mask_array.shape[3]]))
            return mask_array
        elif self.dict_masking.multimod_type == 'and':
            for t in range(0, max_time):
                new_mask = np.ones(data_array.shape[0:3])
                for mod in mod_masking:
                    if np.count_nonzero(data_array[..., mod, t]) > 0:
                        new_mask = np.multiply(new_mask,
                                               mask_array[..., mod, t])
                mask_array[..., t:t + 1] = io.expand_for_5d(np.tile(
                    np.expand_dims(new_mask, axis=3), [1, mask_array.shape[3]]))
            return mask_array
        else:
            return mask_array

    def normalise_subject_data(self, data_dict):
        """
        Call this function to normalise the subject already loaded data.
        """
        data_dict.input = np.nan_to_num(data_dict.input)
        mask_array = self.make_mask_array(data_dict.input)
        data_dict.input = self.standardisor.normalise_data_array(
            data_dict.input, mask_array)
        return data_dict

    def next_subject(self, do_shuffle=True):
        """
        Call this function to get the next subject's image data.
        """
        # go to the next subject in the list (avoid running out of the list)
        self.current_id = (self.current_id + 1) % len(self.subject_list)
        if do_shuffle:
            shuffle(self.subject_list)
        current_subject = self.subject_list[self.current_id]
        data_dict = current_subject.read_all_modalities(
            self.flags.flag_reorient, self.flags.flag_resample)

        if self.flags.flag_standardise and not self.flags.flag_save_norm:
            data_dict = self.normalise_subject_data(data_dict)

        if self.flags.flag_whiten:
            data_dict.input = self.whiten_subject_data_array(data_dict.input)

        return data_dict.input, data_dict.output, data_dict.weight, \
               current_subject

