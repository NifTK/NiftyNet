# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import csv
import os
import sys
from difflib import SequenceMatcher

import numpy as np

from niftynet.io.misc_io import touch_folder
from niftynet.utilities.filename_matching import KeywordsMatching


def match_first_degree(name_list1, name_list2):
    """
    First immediate matching between two possible name lists (exact equality
    between one item of list1 and of list2
    :param name_list1: First list of names to match
    :param name_list2: Second list of names where to find a match
    :return init_match1:
    :return init_match2:
    :return ind_match1: Indices of second list that correspond to each given
    item of list 1 if exists (-1 otherwise)
    :return ind_match2: Indices of first list that correspond to each given
    item of list 2 if exists (-1) otherwise
    """
    if name_list1 is None or name_list2 is None:
        return None, None, None, None
    init_match1 = [''] * len(name_list1)
    init_match2 = [''] * len(name_list2)
    ind_match1 = [-1] * len(name_list1)
    ind_match2 = [-1] * len(name_list2)
    flatten_list1 = [item for sublist in name_list1 for item in sublist]
    flatten_list2 = [item for sublist in name_list2 for item in sublist]
    indflat_1 = [i for i in range(0, len(init_match1)) for item in
                 name_list1[i] if init_match1[i] == '']
    indflat_2 = [i for i in range(0, len(init_match2)) for item in
                 name_list2[i] if init_match2[i] == '']
    for i in range(0, len(name_list1)):
        for name in name_list1[i]:
            if name in flatten_list2:
                init_match1[i] = name
                ind_match1[i] = indflat_2[flatten_list2.index(name)]
                break
    for i in range(0, len(name_list2)):
        for name in name_list2[i]:
            if name in flatten_list1:
                init_match2[i] = name
                ind_match2[i] = indflat_1[flatten_list1.index(name)]
                break
    return init_match1, init_match2, ind_match1, ind_match2


def __find_max_overlap_in_list(name, list_names):
    """
    Given a name and list of names to match it to, find the maximum overlap
    existing

    :param name: string to match to any of list_names
    :param list_names: list of candidate strings
    :return match_seq: matched substring
    :return index: index of element in list_names to which the match is
    associated. Returns -1 if there is no found match
    """
    match_max = 0
    match_seq = ''
    match_orig = ''
    match_ratio = 0
    if not list_names:
        return '', -1
    for test in list_names:
        if test:
            match = SequenceMatcher(None, name, test).find_longest_match(
                0, len(name), 0, len(test))
            if match.size >= match_max \
                    and match.size / len(test) >= match_ratio:
                match_max = match.size
                match_seq = test[match.b:(match.b + match.size)]
                match_ratio = match.size / len(test)
                match_orig = test
    if match_max == 0:
        return '', -1
    other_list = [name for name in list_names
                  if match_seq in name and match_max / len(name) == match_ratio]
    if len(other_list) > 1:
        return '', -1
    return match_seq, list_names.index(match_orig)


def match_second_degree(name_list1, name_list2):
    """
    Perform the double matching between two lists of
    possible names.
    First find the direct matches, remove them from
    the ones still to match and
    match the remaining ones using the maximum overlap.
    Returns the name
    match for each list, and the index correspondences.

    More subtle matching with first direct matching and then secondary
    overlap matching between list of list of potential names
    :param name_list1:
    :param name_list2:
    :return init_match1:
    :return ind_match1: Index of corresponding match in name_list2
    :return init_match2: Matching string in list2
    :return ind_match2: Index of corresponding match in name_list1
    """
    if name_list1 is None or name_list2 is None:
        return None, None, None, None
    init_match1, init_match2, ind_match1, ind_match2 = match_first_degree(
        name_list1, name_list2)
    reduced_list1 = [names for names in name_list1
                     if init_match1[name_list1.index(names)] == '']
    reduced_list2 = [names for names in name_list2
                     if init_match2[name_list2.index(names)] == '']
    redflat_1 = [item for sublist in reduced_list1 for item in sublist]
    indflat_1 = [i for i in range(0, len(init_match1)) for item in
                 name_list1[i] if init_match1[i] == '']
    redflat_2 = [item for sublist in reduced_list2 for item in sublist]
    indflat_2 = [i for i in range(0, len(init_match2)) for item in
                 name_list2[i] if init_match2[i] == '']
    for i in range(0, len(name_list1)):
        if init_match1[i] == '':
            for n in name_list1[i]:
                init_match1[i], index = __find_max_overlap_in_list(n, redflat_2)
                if index >= 0:
                    ind_match1[i] = indflat_2[index]
    for i in range(0, len(name_list2)):
        if init_match2[i] == '':
            for n in name_list2[i]:
                init_match2[i], index = __find_max_overlap_in_list(n, redflat_1)
                if index >= 0:
                    ind_match2[i] = indflat_1[index]
    return init_match1, ind_match1


# From a list of list of names and a list of list of files that are
# associated, find the name correspondence and therefore the files associations
def join_subject_id_and_filename_list(name_list, list_files):
    """
    From the list of list of names and the list of list of files
    corresponding to each constraint find the association between a single
    name id and the different file lists
    :param name_list: list of list of names
    :param list_files: list of list of files (one list per constraint)
    :return list_combined: List per subject of name and list of files given
    by the constraints
    """
    ind_max = np.argmax([len(names) for names in name_list])
    name_max = name_list[ind_max]
    name_tot = []
    ind_tot = []
    name_max_to_use = []
    for c in range(0, len(list_files)):
        name_match, ind_match = match_second_degree(name_max, name_list[c])
        if c == ind_max:
            name_max_to_use = name_match
        name_tot.append(name_match)
        ind_tot.append(ind_match)

    list_combined = []
    for (i, name) in enumerate(name_max_to_use):
        list_temp = [name]
        # To do : Taking care of the case when the list of a constraint is
        # completely empty
        for c in range(0, len(list_files)):
            output = list_files[c][ind_tot[c][i]] if ind_tot[c][i] > -1 else ''
            list_temp.append(output)
        list_combined.append(list_temp)
    return list_combined


def remove_duplicated_names(name_list):
    """
    From a list of list of names remove the items that are duplicated
    :param name_list: list of list of names to investigate
    :return duplicates_removed: list of list of names freed of duplicates
    """
    flattened_list = [item for sublist in name_list for item in sublist]
    list_duplicated = [item for item in flattened_list
                       if flattened_list.count(item) > 1]
    duplicates_removed = []
    for names in name_list:
        duplicates_removed.append([name for name in names
                                   if name not in list_duplicated])
    return duplicates_removed


def write_csv(csv_file, list_combined):
    # csv writer has different behaviour in python 2/3
    if sys.version_info[0] >= 3:
        with open(csv_file, 'w', newline='', encoding='utf8') as csvfile:
            file_writer = csv.writer(csvfile)
            for list_temp in list_combined:
                file_writer.writerow(list_temp)
    else:
        with open(csv_file, 'wb') as csvfile:
            file_writer = csv.writer(csvfile, delimiter=',')
            for list_temp in list_combined:
                file_writer.writerow(list_temp)
    return


def match_and_write_filenames_to_csv(list_constraints, csv_file):
    """
    Combine all elements of file searching until finally writing the names
    :param list_constraints: list of constraints (defined by list of paths to
    search, list of elements the filename should contain and of those that
    are forbidden
    :param csv_file: file on which to write the final list of files.
    :return:
    """
    name_tot = []
    list_tot = []
    if list_constraints is None or len(list_constraints) == 0:
        return
    for c in list_constraints:
        list_files, name_list = \
            KeywordsMatching.matching_subjects_and_filenames(c)
        name_list = remove_duplicated_names(name_list)
        name_tot.append(name_list)
        list_tot.append(list_files)
    list_combined = join_subject_id_and_filename_list(name_tot, list_tot)
    list_combined = filter(lambda names: '' not in names, list_combined)
    list_combined = list(list_combined)
    if not list_combined:
        raise IOError('Nothing to write to {}'.format(csv_file))
    touch_folder(os.path.dirname(csv_file))
    write_csv(csv_file, list_combined)

    return list_combined
