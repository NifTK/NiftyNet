# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import numpy as np
import utilities.misc_io as util


class KeywordsMatching(object):
    def __init__(self, list_paths=(), list_contain=(), list_not_contain=()):
        self.list_paths = list_paths
        self.list_contain = list_contain
        self.list_not_contain = list_not_contain

    @classmethod
    def from_tuple(cls, input_tuple):
        path, contain, not_contain = [], [], []
        for (name, value) in input_tuple:
            if len(value) <= 1 or value == '""':
                continue
            if name == "path_to_search":
                value = value.split(',')
                for path_i in value:
                    path_i = os.path.abspath(path_i.strip())
                    if os.path.exists(path_i):
                        path.append(path_i)
                    else:
                        raise ValueError('folder not found {}'.format(path_i))
            elif name == "filename_contains":
                value = value.split(',')
                for val in value:
                    val = val.strip()
                    contain.append(val)
            elif name == "filename_not_contains":
                value = value.split(',')
                for val in value:
                    val = val.strip()
                    not_contain.append(val)
        path = tuple(set(path))
        contain = tuple(set(contain))
        not_contain = tuple(set(not_contain))
        new_matcher = cls(path, contain, not_contain)
        return new_matcher

    def matching_subjects_and_filenames(self):
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
                name_list_final.append(self.extract_subject_id_from(filename))
        return list_final, name_list_final

    def extract_subject_id_from(self, filename):
        path, name, ext = util.split_filename(filename)

        index_constraint = []
        length_constraint = []
        for c in self.list_contain:
            index_constraint.append(name.find(c))
            length_constraint.append(len(c))
        sort_indices = np.argsort(index_constraint)

        index_init = 0 if index_constraint[sort_indices[0]] > 0 else \
            length_constraint[sort_indices[0]]
        name_pot = []
        index_start = 0 if index_constraint[sort_indices[0]] > 0 else 1
        for i in range(index_start, len(self.list_contain)):
            name_pot_temp = name[index_init: index_constraint[sort_indices[i]]]
            name_pot_temp = ''.join(x for x in name_pot_temp if x.isalnum())
            if name_pot_temp is not '':
                name_pot.append(name_pot_temp)
            index_init = index_constraint[sort_indices[i]] + \
                len(self.list_contain[sort_indices[i]])

        name_temp = name[index_init:]
        name_temp = ''.join(x for x in name_temp if x.isalnum())
        if name_temp is not '':
            name_pot.append(name_temp)
        return name_pot
