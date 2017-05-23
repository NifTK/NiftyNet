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
from utilities.CSVTable import CSVTable


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
                 csv_dict=None,
                 number_list=None,
                 flags=cc.Flags(),
                 loss=['dice']):

        self.dict_normalisation = dict_normalisation
        self.number_list = number_list
        self.flags = flags
        self.loss = loss

        self.csv_table = CSVTable(csv_file, csv_dict)

        self.standardisor = HistNormaliser_bis(
            self.dict_normalisation.hist_ref_file,
            self.dict_normalisation.path_to_train, dict_masking,
            self.dict_normalisation.norm_type, self.dict_normalisation.cutoff,
            dict_masking.mask_type, '')

        self.subject_list = self.create_subject_list()
        self.current_id = -1

    def list_input_filenames_from_subjects(self, subjects):
        if subjects is None:
            return {}
        return [s.file_path_list.input.array_files for s in subjects]

    def create_dict_modalities_from_subjects(self, subjects):
        if subjects is None:
            return {}
        num_modality = subjects[0].file_path_list.input.num_modality
        dict_modalities = {}
        for m in range(0, num_modality):
            name_mod = 'Modality-{}'.format(m)
            dict_modalities[name_mod] = m
        return dict_modalities


    # Provide the final list of eligible subjects
    def create_subject_list(self):

        subjects = self.csv_table.to_subject_list()
        modalities = self.create_dict_modalities_from_subjects(subjects)
        mod_to_train = self.standardisor.check_modalities_to_train(modalities)
        if self.flags.flag_standardise and len(mod_to_train) > 0:
            print("Training normalisation histogram references")
            array_files = self.list_input_filenames_from_subjects(subjects)
            new_mapping = self.standardisor \
                .training_normalisation_from_array_files(
                array_files, mod_to_train)
            self.standardisor.complete_and_transform_model_file(
                new_mapping, mod_to_train.keys())
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

        return data_dict.input, data_dict.output, data_dict.weight, self.current_id

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
