# -*- coding: utf-8 -*-
"""
Matching file names by configuration options.
"""
from __future__ import absolute_import, print_function

import os
import re

import niftynet.io.misc_io as util


class KeywordsMatching(object):
    """
    This class is responsible for the search of the appropriate files to use
    as input based on the constraints given in the config file.
    """

    def __init__(self, list_paths=(), list_contain=(), list_not_contain=()):
        self.path_to_search = list_paths
        self.filename_contains = list_contain
        self.filename_not_contains = list_not_contain

    @classmethod
    def from_tuple(cls, input_tuple, default_folder=None):
        """
        In the config file, constraints for a given search can be of three
        types:
        ``path_to_search``, ``filename_contains`` and
        ``filename_not_contains``. Each associated value is a string.
        Multiple constraints are delimited by a ``,``.
        This function creates the corresponding matching object with the list
        of constraints for each of these subtypes.

        :param default_folder: relative paths are first tested against
            the current folder, and then against this default folder.
        :param input_tuple: set of searching parameters.
        :return:
        """
        path, contain, not_contain = [], (), ()
        for (name, value) in input_tuple:
            if not value:
                continue
            if name == "path_to_search":
                value = value.split(',')
                for path_i in value:
                    path_i = path_i.strip()
                    path_orig = os.path.abspath(path_i)
                    if os.path.exists(path_orig):
                        path.append(path_orig)
                        continue

                    if not default_folder:
                        raise ValueError(
                            'data input folder {} not found, did you maybe '
                            'forget to download data?'.format(path_i))
                    path_def = os.path.join(default_folder, path_i)
                    path_def = os.path.abspath(path_def)
                    if not os.path.exists(path_def):
                        raise ValueError(
                            'data input folder {} not found, did you maybe '
                            'forget to download data?'.format(path_i))
                    path.append(path_def)

            elif name == "filename_contains":
                contain = tuple(set(value))
            elif name == "filename_not_contains":
                not_contain = tuple(set(value))
        path = tuple(set(path))
        new_matcher = cls(path, contain, not_contain)
        return new_matcher

    def matching_subjects_and_filenames(self):
        """
        This function perform the search of the relevant files (stored in
        filename_list) and extract
        the corresponding possible list of subject names (stored in
        subjectname_list).

        :return: filename_list, subjectname_list
        """
        path_file = [(p, filename)
                     for p in self.path_to_search
                     for filename in sorted(os.listdir(p))]
        matching_path_file = list(filter(self.__is_a_candidate, path_file))
        filename_list = \
            [os.path.join(p, filename) for p, filename in matching_path_file]
        subjectname_list = [self.__extract_subject_id_from(filename)
                            for p, filename in matching_path_file]
        if not filename_list or not subjectname_list:
            raise IOError('no file matched based on this matcher: {}'.format(
                self.__dict__))
        return filename_list, subjectname_list

    def __is_a_candidate(self, x):
        all_pos_match = all(c in x[1] for c in self.filename_contains)
        all_neg_match = not any(c in x[1] for c in self.filename_not_contains)
        return all_pos_match and all_neg_match

    def __extract_subject_id_from(self, fullname):
        """
        This function returns a list of potential subject names from a given
        filename, knowing the imposed constraints. Constraints strings are
        removed from the filename to provide the list of possible names. If
        after reduction of the filename from the constraints the name is
        empty the initial filename is returned.

        :param fullname:
        :return name_pot: list of potential subject name given the constraint
         list and the initial filename
        """
        _, name, _ = util.split_filename(fullname)
        # split name into parts that might be the subject_id
        noncapturing_regex_delimiters = \
            ['(?:{})'.format(re.escape(c)) for c in self.filename_contains]
        if noncapturing_regex_delimiters:
            potential_names = re.split(
                '|'.join(noncapturing_regex_delimiters), name)
        else:
            potential_names = [name]
        # filter out non-alphanumeric characters and blank strings
        potential_names = [re.sub(r'\W+', '', name) for name in potential_names]
        potential_names = list(filter(bool, potential_names))
        if len(potential_names) > 1:
            potential_names.append(''.join(potential_names))
        return potential_names
