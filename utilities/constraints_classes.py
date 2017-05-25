import os

import numpy as np

import misc_io as util


class Normalisation(object):
    def __init__(self, path_to_save, hist_ref_file, norm_type='percentile',
                 cutoff=[0.05, 0.95], flag_saving=True, list_train_dir=[]):
        self.path_to_save = path_to_save
        self.norm_type = norm_type
        self.cutoff = cutoff
        self.hist_ref_file = hist_ref_file
        self.flag_saving = flag_saving
        self.path_to_train = list_train_dir

    def _update_dict_normalisation(self, param):
        self.path_to_save = param.saving_norm_dir
        self.norm_type = param.norm_type
        self.cutoff = [x for x in param.norm_cutoff]
        self.hist_ref_file = param.histogram_ref_file
        self.flag_saving = (param.flag_saving_norm == 'True')
        # TODO change default path_to_save for training
        self.path_to_train = [p for p in param.train_data_dir.split(' ')]
        if len(self.path_to_save) == 0:
            self.path_to_save = os.path.join(
                self.path_to_train[0], 'Normalising')


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
                 multimod_type='or',
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


class ConstraintSearch(object):
    def __init__(self, list_paths=[], list_contain=[], list_not_contain=[],
                 list_clean=[]):
        self.list_paths = list_paths
        self.list_contain = list_contain
        self.list_not_contain = list_not_contain
        self.list_clean = list_clean

    def create_list_from_constraint(self):
        list_final = []
        name_list_final = []
        for p in self.list_paths:
            for filename in os.listdir(p):
                if any(c not in filename for c in self.list_contain):
                    continue
                if any(c in filename for c in self.list_not_contain):
                    continue
                full_file_name = os.path.join(p, filename)
                list_final.append(full_file_name)
                name_list_final.append(self.list_subjects_potential(filename))
        return list_final, name_list_final

    def list_subjects_potential(self, filename):
        index_constraint = []
        length_constraint = []
        path, name, ext = util.split_filename(filename)
        for c in self.list_contain:
            index_constraint.append(name.find(c))
            length_constraint.append(len(c))
        sort_indices = np.argsort(index_constraint)

        index_init = 0 if index_constraint[sort_indices[0]] > 0 else \
            length_constraint[sort_indices[0]]
        name_pot = []
        index_start = 0 if index_constraint[sort_indices[0]] > 0 else 1
        for i in range(index_start, len(self.list_contain)):
            name_pot_temp = name[
                            index_init: index_constraint[sort_indices[i]]]
            for c in self.list_clean:
                if c in name_pot_temp:
                    name_pot_temp = name_pot_temp.replace(c, '')
            name_pot.append(name_pot_temp)
            index_init = index_constraint[sort_indices[i]] + \
                         len(self.list_contain[sort_indices[i]])
        name_pot.append(name[index_init:])
        return name_pot
