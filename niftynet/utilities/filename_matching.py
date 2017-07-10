# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os,re

import numpy as np
import niftynet.utilities.misc_io as util


class KeywordsMatching(object):
    '''
    This class is responsible for the search of the appropriate files to use
    as input based on the constraints given in the config file
    '''
    def __init__(self, list_paths=(), list_contain=(), list_not_contain=()):
        self.list_paths = list_paths
        self.list_contain = list_contain
        self.list_not_contain = list_not_contain

    @classmethod
    def from_tuple(cls, input_tuple):
        '''
        In the config file, constraints for a given search can be of three
        types:
        path_to_search, filename_contains and filename_not_contains. Each
        associated value is a string. Multiple constraints are delimited by a ,
        This function creates the corresponding matching object with the list
        of constraints for each of these subtypes.
        :param input_tuple:
        :return:
        '''
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
                        raise ValueError('data input folder {} not found, did'
                                         ' you maybe forget to download data?'
                                         .format(path_i))
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
        ''''
        This function perform the search of the relevant files (stored in
        list_final) and extract
        the corresponding possible list of subject names (stored in
        name_list_final).
        :returns list_final, name_list_final
        '''
        path_file=[(p,filename) for p in self.list_paths for filename in os.listdir(p)]
        func_match = lambda x:     all(c in x[1] for c in self.list_contain) and not any(c in x[1] for c in self.list_not_contain)
        matching_path_file=list(filter(func_match,path_file))
        list_final=[os.path.join(p,filename) for p,filename in matching_path_file]
        name_list_final=[self.extract_subject_id_from(filename) for p,filename in matching_path_file]
        return list_final, name_list_final

    def extract_subject_id_from(self, filename):
        '''
        This function returns a list of potential subject names from a given
        filename, knowing the imposed constraints. Constraints strings are
        removed from the filename to provide the list of possible names. If
        after reduction of the filename from the constraints the name is
        empty the initial filename is returned.
        :param filename:
        :return name_pot: list of potential subject name given the constraint
         list and the initial filename
        '''
        path, name, ext = util.split_filename(filename)
        # split name into parts that might be the subject_id
        noncapturing_regex_delimiters=['(?:'+re.escape(c)+')' for c in self.list_contain]
        potential_names=re.split('|'.join(noncapturing_regex_delimiters),name)
        # filter out non-alphanumeric characters and blank strings
        potential_names=[re.sub(r'\W+', '', name) for name in potential_names]
        potential_names=[name for name in potential_names if name is not '']
        return potential_names

