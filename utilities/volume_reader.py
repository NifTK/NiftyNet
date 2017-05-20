import os
from random import shuffle

import numpy as np

from utilities.subject import Subject
import utilities.misc_grep_file as misc_grep
import nn.histogram_standardisation as hs
from nn.preprocess import HistNormaliser_bis

ORNT_STAND = [[0, 1], [1, 1], [2, 1]]


class VolumePreprocessor(object):
    """
    This class manages the loading step, i.e., return subject's data
    by searching user provided path and modality constraints.
    The volumes are resampled/reoriented if required.

    This class maintains a list of subjects, where each element of the list
    is a Patient object.
    """

    def __init__(self,
                 dict_constraint,
                 dict_normalisation,
                 dict_masking,
                 csv_file=None,
                 list_nearest_neighbours=[],
                 flag_save_norm=False,
                 flag_standardisation=True,
                 flag_whitening=True,
                 do_reorient=False,
                 do_resample=False):

        self.list_paths_to_search = dict_constraint.path_to_search
        self.comp_mod_list = dict_constraint.mod_comp
        self.opt_mod_list = dict_constraint.mod_optional
        self.comp_seg_list = dict_constraint.seg_comp
        self.opt_seg_list = dict_constraint.seg_optional
        self.comp_weight_list = dict_constraint.weight_comp
        self.opt_weight_list = dict_constraint.weight_optional
        self.list_nearest_neighbours = list_nearest_neighbours
        self.do_reorient = dict_constraint.flag_orientation
        self.do_resample = dict_constraint.flag_isotropic
        self.standardisor = HistNormaliser_bis(dict_normalisation.hist_ref_file,
                                               self.list_paths_to_search)
        self.hist_ref_file = dict_normalisation.hist_ref_file
        self.dict_masking = dict_masking
        self.dict_normalisation = dict_normalisation
        self.dict_constraint = dict_constraint
        self.csv_file = csv_file
        self.flag_standardisation = flag_standardisation
        self.flag_save_norm = dict_normalisation.flag_saving
        self.flag_whitening = flag_whitening

        self.subject_list = self._search_for_eligible_subjects()
        self.current_id = 0

    # Look and create initial subjects according to constraints.
    def search_with_constraints(self):
        subjects = []
        list_constraints = self.comp_mod_list + self.comp_seg_list + \
                           self.comp_weight_list
        list_subjects_constrained = misc_grep.combine_list_subjects_compulsory(
            self.list_paths_to_search, list_constraints)
        for name in list_subjects_constrained:
            dict_available = misc_grep.create_list_modalities_available(
                self.list_paths_to_search, name)
            opt_mod_available = [mod for mod in self.opt_mod_list if
                                 len(dict_available[mod]) > 0]
            opt_seg_available = []
            opt_weight_available = []
            for s in dict_available['other']:
                for seg in self.comp_seg_list:
                    dict_available[seg] = [s]
                for weight in self.comp_weight_list:
                    dict_available[weight] = [s]
                for seg in self.opt_seg_list:
                    if seg in s:
                        opt_seg_available.append(seg)
                        dict_available[seg] = [s]
                for weight in self.opt_weight_list:
                    if weight in s:
                        opt_weight_available.append(weight)
                        dict_available[weight] = [s]
            seg_list = self.comp_seg_list + opt_seg_available
            mod_list = self.comp_mod_list + opt_mod_available
            weight_list = self.comp_weight_list + opt_weight_available

            modality_list = mod_list + seg_list + weight_list
            list_nn = [mod for mod in
                       modality_list if mod in self.list_nearest_neighbours]
            file_path_dict = {mod: dict_available[mod][0] for mod in
                              modality_list}
            subject = Subject(name=name,
                              file_path_dict=file_path_dict,
                              list_nn=list_nn)
            subjects.append(subject)
        return subjects

    # Provide the final list of eligible subjects
    def _search_for_eligible_subjects(self):
        subjects = []
        modalities = []
        if not self.csv_file:
            subjects = self.search_with_constraints()
            modalities = self.comp_mod_list + self.opt_mod_list
        # else:
        #     subjects = self.list_with_csv() # Not implemented yet
        # Perform the standardisation on the subjects
        if self.flag_standardisation and self.flag_save_norm:
            if len(hs.check_modalities_to_train(self.hist_ref_file,
                                                modalities)) > 0:
                raise ValueError("The histogram reference file is not "
                                 "complete, please run training histogram "
                                 "before hand")
            else:
                for s in subjects:
                    self.normalise_subject_data_and_save(s)

        return subjects

    def whiten_subject_data(self, data_dict, modalities):
        list_mod_whiten = [m for m in modalities if m in data_dict.keys()]
        mask_dict = self.make_mask_dict(data_dict)
        for m in list_mod_whiten:
            data_dict[m] = self.standardisor.whitening_transformation(
                data_dict[m], mask_dict[m])
        return data_dict

    def normalise_subject_data_and_save(self, subject):
        if self.flag_standardisation:
            data_dict = subject.read_all_modalities(self.dict_constraint.flag_orientation,
                                                    self.dict_constraint.flag_isotropic)
            mask_dict = self.make_mask_dict(data_dict)
            data_dict = self.standardisor.normalise_data(data_dict, mask_dict)
            for mod in mask_dict.keys():
                name_norm_save = io.create_new_filename(
                    subject.file_path_dict[mod],
                    new_path=self.dict_normalisation.path_to_save,
                    new_prefix='Norm')
                io.save_img(data_dict[mod], subject.name, [], name_norm_save,
                            filename_ref=subject.file_path_dict[mod],
                            flag_orientation=self.do_reorient,
                            flag_isotropic=self.do_resample)
                subject._set_data_path(name_norm_save, mod)

    def make_mask_dict(self, data_dict):
        mask_dict = {}
        list_modalities = self.comp_mod_list + self.opt_mod_list
        list_modalities = [m for m in list_modalities if m in data_dict.keys()]
        if not self.dict_masking.multimod_type == 'and' and not \
                        self.dict_masking.multimod_type == 'or':
            mod_masking = [m for m in list_modalities]
        else:
            mod_masking = [m for m in self.dict_masking.multimod if m in
                           list_modalities]
        for mod in mod_masking:
            new_mask = hs.create_mask_img_multimod(
                data_dict[mod], self.dict_masking.mask_type)
            mask_dict[mod] = util.adapt_to_shape(new_mask, data_dict[mod].shape)
        if self.dict_masking.multimod_type == 'or':
            new_mask = np.zeros(data_dict[self.comp_mod_list[0]].shape)
            for mod in mod_masking:
                new_mask = new_mask + mask_dict[mod]
            new_mask[new_mask > 0.5] = 1
            for mod in list_modalities:
                mask_dict[mod] = new_mask
            return mask_dict
        elif self.dict_masking.multimod_type == 'and':
            new_mask = np.ones(data_dict[self.comp_mod_list[0]].shape)
            for mod in mod_masking:
                new_mask = np.multiply(new_mask, mask_dict[mod])
            for mod in list_modalities:
                mask_dict[mod] = new_mask
            return mask_dict
        else:
            return mask_dict

    def normalise_subject_data(self, data_dict):
        """
        Call this function to normalise the subject already loaded data.
        """
        mask_dict = self.make_mask_dict(data_dict)
        data_dict = self.standardisor.normalise_data(data_dict, mask_dict)
        return data_dict

    def next_subject(self, do_shuffle=True):
        """
        Call this function to get the next subject's image data.
        """
        # go to the next subject in the list (avoid running out of the list)
        self.current_id = (self.current_id + 1) % len(self.subject_list)
        if do_shuffle:
            shuffle(self.subject_list)
            # self.subject_list.shuffle()
        current_subject = self.subject_list[self.current_id]
        data_dict = current_subject.read_all_modalities(self.do_reorient,
                                                        self.do_resample)
        if self.flag_standardisation and not self.flag_save_norm:
            data_dict = self.normalise_subject_data(data_dict)

        if self.flag_whitening:
            modalities_to_whiten = self.comp_mod_list + self.opt_mod_list
            data_dict = self.whiten_subject_data(data_dict, modalities_to_whiten)

        img, seg, weights = self.data_distribution(current_subject, data_dict)
        return img, seg, weights, current_subject

    def data_distribution(self, subject, data):
        """
        This function redistributes the data dictionary into usable 4D (to
        be 5D) images into img, seg and weightsmap it follows the order given
        by the constraint (compulsory first, optional afterwards,
        image modalities before segmentations
        """
        list_img = [mod for mod in subject.file_path_dict.keys() if mod in
                    self.comp_mod_list + self.opt_mod_list]
        list_seg = [seg for seg in subject.file_path_dict.keys() if seg in
                    self.comp_seg_list + self.opt_seg_list]
        list_weight = [weight for weight in subject.file_path_dict.keys() if
                       weight in self.comp_weight_list + self.opt_weight_list]
        img = None
        seg = None
        weights = None
        for i in list_img:
            img_data = data[i]
            if img_data.ndim == 3:
                img_data = np.expand_dims(img_data, axis=3)
            if img is None:
                img = img_data
            else:
                img = np.concatenate([img, img_data], axis=3)
        for s in list_seg:
            seg_data = data[s]
            if seg_data.ndim == 3:
                seg_data = np.expand_dims(seg_data, axis=3)
            if seg is None:
                seg = seg_data
            else:
                seg = np.concatenate([seg, seg_data], axis=3)
        for w in list_weight:
            weight_data = data[w]
            if weight_data.ndim == 3:
                weight_data = np.expand_dims(weight_data, axis=3)
            if weights is None:
                weights = weight_data
            else:
                weights = np.concatenate([weights, weight_data], axis=3)
        return img, seg, weights


class Constraints(object):
    def __init__(self,
                 list_path, mod_comp, seg_comp, weight_comp, mod_optional,
                 seg_optional, weight_optional, orientation, isotropic):
        self.path_to_search = list_path
        self.mod_comp = mod_comp
        self.seg_comp = seg_comp
        self.weight_comp = weight_comp
        self.mod_optional = mod_optional
        self.seg_optional = seg_optional
        self.weight_optional = weight_optional
        self.flag_orientation = orientation
        self.flag_isotropic = isotropic

    def _update_dict_constraint(self, param):
        path_to_search = []
        seg_comp = []
        weight_comp = []
        if param.action == 'eval':
            self.path_to_search = [x for x in param.eval_data_dir.split(' ')]
        else:
            self.path_to_search = [x for x in param.train_data_dir.split(' ')]
        self.seg_comp = [x for x in param.seg_compulsory.split(' ')]
        self.weight_comp = [x for x in param.weight_compulsory.split(' ') if
                            len(
                                x) > 1]
        self.mod_comp = [x for x in
                         param.mod_compulsory.split(' ')]
        self.mod_optional = [x for x in param.mod_optional.split(' ') if
                             len(x) > 1]
        self.seg_optional = [x for x in param.seg_optional.split(' ') if
                             len(x) > 1]
        self.weight_comp = weight_comp
        self.weight_optional = [x for x in
                                param.weight_optional.split(
                                    ' ')
                                if len(x) > 1]
        self.flag_isotropic = (
            param.flag_isotropic == 'True')
        self.flag_orientation = (
            param.flag_orientation == 'True')


class Normalisation(object):
    def __init__(self, path_to_save, hist_ref_file, norm_type='percentile',
                 cutoff=[0.05, 0.95], flag_saving=True, list_train_dir=[]):
        self.path_to_save = path_to_save
        self.norm_type = norm_type
        self.cutoff = cutoff
        self.hist_ref_file = hist_ref_file
        self.flag_saving = flag_saving
        self.list_train_dir = list_train_dir

    def _update_dict_normalisation(self, param):
        self.path_to_save = param.saving_norm_dir
        self.norm_type = param.norm_type
        self.cutoff = [x for x in param.norm_cutoff]
        self.hist_ref_file = param.histogram_ref_file
        self.flag_saving = (param.flag_saving_norm == 'True')
        # TODO change default path_to_save for training
        list_train_dir = [p for p in param.train_data_dir.split(' ')]
        if len(self.path_to_save) == 0:
            self.path_to_save = os.path.join(
                list_train_dir[0], 'Normalising')


class Augmentation(object):
    def __init__(self, rotation=False, spatial_scaling=False):
        self.rotation = rotation
        self.spatial_scaling = spatial_scaling

    def _update_dict_preprocess(self, param):
        self.rotation = (param.flag_rotation == 'True')
        self.spatial_scaling = (param.flag_spatial_scaling == 'True')
        if param.action == "inference":
            self.rotation = False
            self.spatial_scaling = False


class Masking(object):
    def __init__(self, mask_type='otsu_plus', path_to_save='',
                 multimod_type='and',
                 multimod=[0],
                 alpha=0.,
                 flag_saving=False, val=0, ):
        self.mask_type = mask_type
        self.path_to_save = path_to_save
        self.multimod_type = multimod_type
        self.multimod = multimod
        self.alpha = alpha
        self.flag_saving = flag_saving
        self.val = val

    def _update_dict_masking(self, param):
        self.multimod = [x for x in param.mask_multimod.split(' ')],
        self.multimod_type = param.mask_multimod_type
        self.mask_type = param.mask_type
        self.flag_saving = (param.flag_saving_mask == 'True')
        self.path_to_save = param.saving_mask_dir
        self.val = 0
        # TODO change default path_to_save for training
        list_train_dir = [p for p in param.train_data_dir.split(' ')]
        if self.path_to_save is None or len(
                self.path_to_save) == 0:
            self.path_to_save = os.path.join(list_train_dir[0], 'Masking')


class Sampling(object):
    def __init__(self, compulsory_labels=[0], minimum_ratio=[0],
                 min_numb_labels=[0],
                 sampling_dim=3):
        self.compulsory_labels = compulsory_labels
        self.minimum_ratio = minimum_ratio
        self.min_numb_labels = min_numb_labels
        self.sampling_dim = sampling_dim

    def _update_dict_sampling(self, param):
        comp_labels = [x for x in
                       map(int, param.compulsory_labels.split(' ' ''))]
        self.compulsory_labels = comp_labels
        self.minimum_ratio = param.min_sampling_ratio
        self.min_numb_labels = param.min_numb_labels
        self.sampling_dim = 3

#
# # To use these classes:
# volume_reader = VolumeReader(param.path_to_search,
#                              .T1', 'T2'],
#                              ['Flair', 'T1c'])
# def sampler():
#     for i in range(iterations):
#         img, seg, weight_map = volume_reader.next_subject()
#         # then we can do preprocessing/random patch sampling
