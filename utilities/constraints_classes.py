import os

import numpy as np

import misc_io as util


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
                name_pot_temp.rsplit(c)
            name_pot.append(name_pot_temp)
            index_init = index_constraint[sort_indices[i]] + \
                         len(self.list_contain[sort_indices[i]])

        name_temp = name[index_init:]
        for c in self.list_clean:
            if c in name_temp:
                name_temp = name_temp.rstrip(c)
                name_temp = name_temp.lstrip(c)
                index_clean = name_temp.find(c)
                if index_clean == 0:
                    name_temp = name_temp[len(c):]
        name_pot.append(name_temp)
        return name_pot

