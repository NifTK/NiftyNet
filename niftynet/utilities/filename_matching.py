# -*- coding: utf-8 -*-
"""
Matching file names by configuration options.
"""
from __future__ import absolute_import, print_function

import os
import re

import six

import niftynet.io.misc_io as util
import tensorflow as tf


class KeywordsMatching(object):
    """
    This class is responsible for the search of the appropriate files to use
    as input based on the constraints given in the config file.
    """

    def __init__(self,
                 list_paths=(),
                 list_contain=(),
                 list_not_contain=(),
                 regex_remove=()):
        self.path_to_search = list_paths
        self.filename_contains = list_contain
        self.filename_not_contains = list_not_contain
        self.filename_toremove_fromid = regex_remove

    @classmethod
    def from_dict(cls, input_dict, default_folder=None):
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
        :param input_dict: set of searching parameters.
        :return:
        """
        path, contain, not_contain, regex = [], (), (), ()
        for (name, value) in input_dict.items():
            if not value:
                continue
            if name == "path_to_search":
                try:
                    # for a string of comma separated path.
                    value = value.split(',')
                except AttributeError:
                    pass

                for path_i in value:
                    path_i = path_i.strip()
                    path_orig = os.path.abspath(os.path.expanduser(path_i))
                    if os.path.exists(path_orig):
                        path.append(path_orig)
                        continue

                    if not default_folder:
                        tf.logging.fatal(
                            'data input folder "%s" not found, did you maybe '
                            'forget to download data?', path_i)
                        raise ValueError
                    path_def = os.path.join(default_folder, path_i)
                    path_def = os.path.abspath(path_def)
                    if not os.path.exists(path_def):
                        tf.logging.fatal(
                            'data input folder "%s" not found, did you maybe '
                            'forget to download data?', path_i)
                        raise ValueError
                    path.append(path_def)

            elif name == "filename_contains":
                contain = tuple(set(value)) \
                    if not isinstance(value, six.string_types) \
                    else tuple([value])
            elif name == "filename_not_contains":
                not_contain = tuple(set(value)) \
                    if not isinstance(value, six.string_types) \
                    else tuple([value])
            elif name == "filename_removefromid":
                regex = tuple(set(value)) \
                    if not isinstance(value, six.string_types) \
                    else tuple([value])
        path = tuple(set(path))
        new_matcher = cls(path, contain, not_contain, regex[0] if regex else "")
        return new_matcher

    def matching_subjects_and_filenames(self):
        """
        This function perform the search of the relevant files (stored in
        filename_list) and extract
        the corresponding possible list of subject names (stored in
        subjectname_list).

        :return: filename_list, subjectname_list
        """
        path_file = [
            (p, filename) for p in self.path_to_search
            for filename in sorted(os.listdir(p))]
        matching_path_file = list(filter(self.__is_a_candidate, path_file))
        filename_list = \
            [os.path.join(p, filename) for p, filename in matching_path_file]
        subjectname_list = [self.__extract_subject_id_from(filename)
            for p, filename in matching_path_file]
        for sname, fname in zip(subjectname_list, filename_list):
            if not sname:
                subjectname_list.remove(sname)
                filename_list.remove(fname)
        self.__check_unique_names(filename_list, subjectname_list)
        if not filename_list or not subjectname_list:
            tf.logging.fatal('no file matched based on this matcher: %s', self)
            raise ValueError
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
        if remove is not empty, will remove only the strings indicated in
        remove. Otherwise, by default will remove all those in filename_contains

        :param fullname:
        :return name_pot: list of potential subject name given the constraint
         list and the initial filename
        """
        _, name, _ = util.split_filename(fullname)
        # split name into parts that might be the subject_id
        potential_names = [name]
        if not self.filename_toremove_fromid:
            #  regular expression not specified,
            #   removing the matched file_contains keywords
            #   use the rest of the string as subject id
            noncapturing_regex_delimiters = \
                ['(?:{})'.format(re.escape(c)) for c in self.filename_contains]
            if noncapturing_regex_delimiters:
                potential_names = re.split(
                    '|'.join(noncapturing_regex_delimiters), name)
            # filter out non-alphanumeric characters and blank strings
            potential_names = [
                re.sub(r'\W+', '', name) for name in potential_names]
        else:
            potential_names = [
                re.sub(self.filename_toremove_fromid, "", name)]
        potential_names = list(filter(bool, potential_names))
        if len(potential_names) > 1:
            potential_names.append(''.join(potential_names))
        return potential_names

    def __check_unique_names(self, file_list, id_list):
        uniq_dict = dict()
        for idx, subject_id in enumerate(id_list):
            if not subject_id:
                continue
            id_string = subject_id[0]
            if id_string in uniq_dict:
                tf.logging.fatal(
                    'extracted the same unique_id "%s" from '
                    'filenames "%s" and "%s", using matcher: %s',
                    id_string, uniq_dict[id_string], file_list[idx], self)
                raise ValueError
            uniq_dict[id_string] = file_list[idx]

    def __str__(self):
        return self.to_string()

    def to_string(self):
        """
        Formatting the class as an intuitive string for printing.

        :return:
        """
        summary_str = '\n\nMatch file names and extract subject_ids from: \n'
        if self.path_to_search:
            summary_str += '-- path to search: {}\n'.format(self.path_to_search)
        if self.filename_contains:
            summary_str += '-- filename contains: {}\n'.format(
                self.filename_contains)
        if self.filename_not_contains:
            summary_str += '-- filename not contains: {}\n'.format(
                self.filename_not_contains)
        if self.filename_toremove_fromid:
            summary_str += '-- filename to remove from id: {}\n'.format(
                self.filename_toremove_fromid)
        return summary_str
