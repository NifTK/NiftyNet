# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os,re

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
        path_file=[(p,filename) for p in self.list_paths for filename in os.listdir(p)]
        func_match = lambda x:     all(c in x[1] for c in self.list_contain) and not any(c in x[1] for c in self.list_not_contain)
        matching_path_file=list(filter(func_match,path_file))
        list_final=[os.path.join(p,filename) for p,filename in matching_path_file]
        name_list_final=[self.extract_subject_id_from(filename) for p,filename in matching_path_file]
        return list_final, name_list_final

    def extract_subject_id_from(self, filename):
        path, name, ext = util.split_filename(filename)
        # split name into parts that might be the subject_id
        noncapturing_regex_delimiters=['(?:'+re.escape(c)+')' for c in self.list_contain]
        potential_names=re.split('|'.join(noncapturing_regex_delimiters),name)
        # filter out non-alphanumeric characters and blank strings
        potential_names=[re.sub(r'\W+', '', name) for name in potential_names]
        potential_names=[name for name in potential_names if name is not '']
        return potential_names