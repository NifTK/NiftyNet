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
        self.csv_list = csv_list
        self.csv_file = csv_file
        self.number_list = number_list
        self.flags = flags
        self.loss = loss

        self.standardisor = HistNormaliser_bis(
            self.dict_normalisation.hist_ref_file,
            self.dict_normalisation.path_to_train, dict_masking,
            self.dict_normalisation.norm_type, self.dict_normalisation.cutoff,
            dict_masking.mask_type, '')

        self.subject_list = self._search_for_eligible_subjects()
        self.current_id = -1

    def create_list_subject_from_csv(self):
        list_subjects = []
        interp_order = self.guess_interp_from_loss()
        interp_order_fields = cc.InputList([3], interp_order, [3], None, None)
        with open(self.csv_file, "rb") as infile:
            reader = csv.reader(infile)
            for row in reader:
                input = cc.CSVCell([[row[1]]])
                output = cc.CSVCell([[row[2]]])
                weight = cc.CSVCell([[row[3]]])
                data_list = cc.InputList(input, output, weight, row[4], row[5])
                new_subject = Subject(row[0],
                                      data_list,
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
            self.csv_list.input,
            self.number_list.input,
            self.flags.flag_allow_missing)
        subjects_output, files_output = misc_csv.create_array_files_from_csv(
            self.csv_list.output,
            self.number_list.output,
            self.flags.flag_allow_missing)
        subjects_weight, files_weight = misc_csv.create_array_files_from_csv(
            self.csv_list.weight,
            self.number_list.weight,
            self.flags.flag_allow_missing)
        subjects_input_txt, files_input_txt = \
            misc_csv.create_array_files_from_csv(self.csv_list.input_txt,
                                                 self.number_list.input_txt,
                                                 self.flags.flag_allow_missing)
        subjects_output_txt, files_output_txt = \
            misc_csv.create_array_files_from_csv(self.csv_list.output_txt,
                                                 self.number_list.output_txt,
                                                 self.flags.flag_allow_missing)
        name_list = cc.InputList(subjects_input,
                                 subjects_output,
                                 subjects_weight,
                                 subjects_input_txt,
                                 subjects_output_txt)
        file_list = cc.InputList(files_input,
                                 files_output,
                                 files_weight,
                                 files_input_txt,
                                 files_output_txt)
        list_combined = misc_csv.combine_list_constraint(name_list, file_list)
        return list_combined

    def create_list_subject_from_list(self):
        interp_order = self.guess_interp_from_loss()
        interp_order_fields = cc.InputList([3], interp_order, [3], None, None)
        list_combined = self.create_array_subjects_csv_list()
        subjects = []
        for s in list_combined:
            input_files = cc.CSVCell(s[1])
            output_files = cc.CSVCell(s[2])
            weight_files = cc.CSVCell(s[3]) if s[3] is not '' else None
            input_txt_files = cc.CSVCell(s[4]) if s[4] is not '' else None
            output_txt_files = cc.CSVCell(s[5]) if s[5] is not '' else None
            file_path_list = cc.InputList(input_files,
                                          output_files,
                                          weight_files,
                                          input_txt_files,
                                          output_txt_files)
            new_subject = Subject(s[0], file_path_list, interp_order_fields)
            subjects.append(new_subject)
        return subjects

    def create_list_array_input_files_from_subjects(self, subjects):
        array_files_tot = []
        if subjects is None:
            subjects = self.create_list_subject_from_csv()
        for s in subjects:
            if not s.file_path_list.input is None:
                array_files_tot.append(s.file_path_list.input.array_files)
        return array_files_tot

    def create_dict_modalities_from_subjects(self, subjects):
        if subjects is None:
            return {}
        subject_ref = subjects[0]
        csv_cell_ref = subject_ref.file_path_list.input
        data_array = io.prepare_5d_data(csv_cell_ref)
        numb_mod = data_array.shape[3]
        dict_modalities = {}
        for m in range(0, numb_mod):
            name_mod = 'Mod%d' % m
            dict_modalities[name_mod] = m
        return dict_modalities


    # Provide the final list of eligible subjects
    def _search_for_eligible_subjects(self):
        modalities = self.dict_normalisation.list_modalities

        if self.csv_file is None and self.csv_list is None:
            raise ValueError("There is not input to build the subjects list!!!")
        if self.csv_file is None:
            subjects = self.create_list_subject_from_list()
        else:
            subjects = self.create_list_subject_from_csv()
        if modalities is None:
            modalities = self.create_dict_modalities_from_subjects(subjects)
        mod_to_train = hs.check_modalities_to_train(
            self.dict_normalisation.hist_ref_file,
            modalities.keys())

        if self.flags.flag_standardise and len(mod_to_train) > 0:
            modalities_to_train = {}
            for mod in mod_to_train:
                modalities_to_train[mod] = modalities[mod]
            warnings.warn("The histogram has to be retrained...")
            array_files = self.create_list_array_input_files_from_subjects(
                subjects)
            #import pdb; pdb.set_trace()
            new_mapping = self.standardisor \
                .training_normalisation_from_array_files(
                array_files, modalities_to_train)
            self.standardisor.complete_and_transform_model_file(
                new_mapping, mod_to_train)
        #if self.flags.flag_standardise and self.flags.flag_save_norm:
        #    for s in subjects:
        #        self.normalise_subject_data_and_save(s)
        return subjects

    def whiten_subject_data_array(self, data_array, modalities_indices=None):
        if modalities_indices is None:
            modalities_indices = range(0, data_array.shape[3])
        list_mod_whiten = [m for m in modalities_indices if
                           m < data_array.shape[3]]
        mask_array = self.standardisor.make_mask_array(data_array)
        for m in list_mod_whiten:
            for t in range(0, data_array.shape[4]):
                data_array[..., m, t] = \
                    self.standardisor.whitening_transformation(
                        data_array[..., m, t], mask_array[..., m, t])
        return data_array

    def whiten_subject_data(self, data_dict, modalities):
        mask_array = self.standardisor.make_mask_array(data_dict.input)
        for m in modalities:
            for t in range(0, len(data_dict[m])):
                data_dict.input[..., m, t] = self.standardisor. \
                    whitening_transformation(
                    data_dict.input[..., m, t], mask_array[..., m, t])
        return data_dict

    #def normalise_subject_data_and_save(self, subject):
    #    if self.flags.flag_standardise:
    #        data_dict = subject.read_all_modalities(self.flags.flag_reorient,
    #                                                self.flags.flag_resample)
    #        data_dict.input = np.nan_to_num(data_dict.input)
    #        mask_array = self.make_mask_array(data_dict.input)
    #        data_dict.input = self.standardisor.normalise_data_array(
    #            data_dict.input, mask_array)
    #        name_norm_save = io.create_new_filename(
    #            subject.name + '.nii.gz',
    #            new_path=self.dict_normalisation.path_to_save,
    #            new_prefix='Norm')
    #        # Put back the array with the nifti conventions.
    #        data_nifti_format = np.swapaxes(data_dict.input, 4, 3)
    #        io.save_img(data_nifti_format, subject.name, [], name_norm_save,
    #                    filename_ref=subject.file_path_list.input.filename_ref,
    #                    flag_orientation=self.flags.flag_reorient,
    #                    flag_isotropic=self.flags.flag_resample)
    #        # TODO: save norm
    #        #subject._set_data_path(name_norm_save)


    def normalise_subject_data(self, data_dict):
        """
        Call this function to normalise the subject already loaded data.
        """
        data_dict.input = np.nan_to_num(data_dict.input)
        mask_array = self.standardisor.make_mask_array(data_dict.input)
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
