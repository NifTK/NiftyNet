# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function,division
import os
import csv
import numpy as np
from difflib import SequenceMatcher
import sys

from niftynet.utilities.filename_matching import KeywordsMatching


# From a unique csv file with for each subject the list of files to use,
# build the 2d array of files to load for each subject and create the overall
# list of such arrays. num_modality indicates the
# number of modalities before going to further time point
# TODO: to support multiple time points
def load_subject_and_filenames_from_csv_file(csv_file,
                                             allow_missing=True,
                                             numb_mod=None):
    '''
    Creates list of names and corresponding file list from csv file
    :param csv_file: file to read built with for each row, the subject name
    followed by a list of files for different time points and modalities.
    :param allow_missing: flag (True/False) indicating if missing modalities
    are possible
    :param numb_mod: number of modalities to consider (needed to account for
    multiple time points
    :return list_subjects: list of subject names
    :return list_filenames: list of filenames for each subject
    '''
    if csv_file is None:
        return [], []
    if not os.path.isfile(csv_file):
        return [], []
    list_subjects = []
    list_filenames = []

    with open(csv_file, "r") as infile:
        reader = csv.reader(infile)
        for row in reader:
            if ('' in row) and (not allow_missing):
                continue
            if ('' in row) and len(set(row[1:])) == 1:
                continue
            subject_name, list_files = [row[0]], row[1:]
            list_subjects.append(subject_name)
            numb_mod = len(list_files) if numb_mod is None else numb_mod
            grouped_time_points = [list_files[i:(i + numb_mod)]
                                   for i in range(0, len(list_files), numb_mod)]
            list_filenames.append(grouped_time_points)
    return list_subjects, list_filenames


# try to find a direct match between two arrays of list of possible names.
def match_first_degree(name_list1, name_list2):
    '''
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
    '''
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


# Find the maximum overlap of n in the list of strings list_names. Returns
# the matched sequence and the corresponding index of the corresponding
# element in the list
def __find_max_overlap_in_list(name, list_names):
    '''
    Given a name and list of names to match it to, find the maximum overlap
    existing
    :param name: string to match to any of list_names
    :param list_names: list of candidate strings
    :return match_seq: matched substring
    :return index: index of element in list_names to which the match is
    associated. Returns -1 if there is no found match
    '''
    match_max = 0
    match_seq = ''
    match_orig = ''
    match_ratio = 0
    if len(list_names) == 0:
        return '', -1
    for test in list_names:
        match = SequenceMatcher(None, name, test).find_longest_match(
            0, len(name), 0, len(test))
        if match.size >= match_max and match.size/len(test) >= \
                match_ratio:
            match_max = match.size
            match_seq = test[match.b:(match.b + match.size)]
            match_ratio = match.size/len(test)
            match_orig = test
    if match_max == 0:
        return '', -1
    other_list = [name for name in list_names if match_seq in name and
                  match_max/len(name) == match_ratio]
    if len(other_list) > 1:
        return '', -1
    return match_seq, list_names.index(match_orig)


# Perform the double matching between two lists of list of possible names.
# First find the direct matches, remove them from the ones still to match and
# match the remaining ones using the maximum overlap. Returns the name
# match for each list, and the index correspondences.
def match_second_degree(name_list1, name_list2):
    '''
    More subtle matching with first direct matching and then secondary
    overlap matching between list of list of potential names
    :param name_list1:
    :param name_list2:
    :return init_match1:
    :return ind_match1: Index of corresponding match in name_list2
    :return init_match2: Matching string in list2
    :return ind_match2: Index of corresponding match in name_list1
    '''
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
    return init_match1, ind_match1, init_match2, ind_match2


# From a list of list of names and a list of list of files that are
# associated, find the name correspondance and therefore the files associations
def join_subject_id_and_filename_list(name_list, list_files):
    '''
    From the list of list of names and the list of list of files
    corresponding to each constraint find the association between a single
    name id and the different file lists
    :param name_list: list of list of names
    :param list_files: list of list of files (one list per constraint)
    :return list_combined: List per subject of name and list of files given
    by the constraints
    '''
    ind_max = np.argmax([len(names) for names in name_list])
    name_max = name_list[ind_max]
    name_tot = []
    ind_tot = []
    name_max_to_use = []
    for c in range(0, len(list_files)):
        name_match, ind_match, _, _ = match_second_degree(name_max, name_list[c])
        if c == ind_max:
            name_max_to_use = name_match
        name_tot.append(name_match)
        ind_tot.append(ind_match)

    list_combined = []
    for (i, name) in enumerate(name_max_to_use):
        list_temp = [name]
        'To do : Taking care of the case when the list of a constraint is ' \
        'completely empty'
        for c in range(0, len(list_files)):
            output = list_files[c][ind_tot[c][i]] if ind_tot[c][i]>-1 else ''
            list_temp.append(output)
        list_combined.append(list_temp)
    return list_combined


def remove_duplicated_names(name_list):
    '''
    From a list of list of names remove the items that are duplicated
    :param name_list: list of list of names to investigate
    :return duplicates_removed: list of list of names freed of duplicates
    '''
    flattened_list = [item for sublist in name_list for item in sublist]
    list_duplicated = [item for item in flattened_list
                       if flattened_list.count(item) > 1]
    duplicates_removed = []
    for names in name_list:
        duplicates_removed.append([name for name in names
                                   if name not in list_duplicated])
    return duplicates_removed


def write_matched_filenames_to_csv(list_constraints, csv_file):
    '''
    Combine all elements of file searching until finally writing the names
    :param list_constraints: list of constraints (defined by list of paths to
    search, list of elements the filename should contain and of those that
    are forbidden
    :param csv_file: file on which to write the final list of files.
    :return:
    '''
    name_tot = []
    list_tot = []
    if list_constraints is None or list_constraints == []:
        return
    for c in list_constraints:
        list_files, name_list = KeywordsMatching.matching_subjects_and_filenames(c)
        name_list = remove_duplicated_names(name_list)
        name_tot.append(name_list)
        list_tot.append(list_files)
    list_combined = join_subject_id_and_filename_list(name_tot, list_tot)
    output_dir = os.path.dirname(csv_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

# def combine_list_constraint(name_list, list_files):
#    name_match_io, ind_io, _, _ = match_second_degree(name_list.input,
#                                                      name_list.output)
#    name_match_iw, ind_iw, _, _ = match_second_degree(name_list.input,
#                                                      name_list.weight)
#    name_match_iit, ind_iit, _, _ = match_second_degree(name_list.input,
#                                                        name_list.input_txt)
#    name_match_iot, ind_iot, _, _ = match_second_degree(name_list.input,
#                                                        name_list.output_txt)
#    if name_list.input is None:
#        raise ValueError("There is no input! Please do check your constraints")
#
#    if not name_list.output is None:
#        list_to_use = name_match_io
#    elif not name_list.weight is None:
#        list_to_use = name_match_iw
#    elif not name_list.input_txt is None:
#        list_to_use = name_match_iit
#    elif not name_list.output_txt is None:
#        list_to_use = name_match_iot
#    else:
#        warnings.warn("You have only an input...")
#        list_temp = remove_duplicated_names(name_list.input)
#        list_to_use = ['_'.join(sublist) for sublist in list_temp]
#
#    list_compare = []
#    for (i, name) in enumerate(list_to_use):
#        input = list_files.input[i]
#        output = list_files.output[ind_io[i]] if ind_io is not None else ''
#        weight = list_files.weight[ind_iw[i]] if ind_iw is not None else ''
#        input_txt = list_files.weight[ind_iit[i]] if ind_iit is not None else ''
#        output_txt = list_files.output_txt[ind_iot[i]] if ind_iot is not None else ''
#        list_temp = [name, input, output, weight, input_txt, output_txt]
#        list_compare.append(list_temp)
#    return list_compare


# def create_csv(constraint_list, csv_file):
#    list_input = None
#    list_output = None
#    list_weight = None
#    list_input_txt = None
#    list_output_txt = None
#    name_output_txt = None
#    name_input_txt = None
#    name_input = None
#    name_output = None
#    name_weight = None
#    if constraint_list.input is not None:
#        list_input, name_input = \
#            KeywordsMatching.create_list_from_constraint(
#                constraint_list.input)
#        name_input = remove_duplicated_names(name_input)
#    if constraint_list.output is not None:
#        list_output, name_output = \
#            KeywordsMatching.create_list_from_constraint(
#                constraint_list.output)
#        name_output = remove_duplicated_names(name_output)
#    if constraint_list.weight is not None:
#        list_weight, name_weight = \
#            KeywordsMatching.create_list_from_constraint(
#                constraint_list.weight)
#        name_weight = remove_duplicated_names(name_weight)
#    if constraint_list.input_txt is not None:
#        list_input_txt, name_input_txt = \
#            KeywordsMatching.create_list_from_constraint(
#                constraint_list.input_txt)
#        name_input_txt = remove_duplicated_names(name_input_txt)
#    if constraint_list.output_txt is not None:
#        list_output_txt, name_output_txt = \
#            KeywordsMatching.create_list_from_constraint(
#                constraint_list.output_txt)
#        name_output_txt = remove_duplicated_names(name_output_txt)
#
#    list_files_init = cc.InputList(list_input, list_output, list_weight,
#                                   list_input_txt, list_output_txt)
#
#    list_names_init = cc.InputList(name_input, name_output, name_weight,
#                                   name_input_txt, name_output_txt)
#
#    list_combined = combine_list_constraint(list_names_init, list_files_init)
#    with open(csv_file, 'wb') as csvfile:
#        file_writer = csv.writer(csvfile, delimiter=',')
#        for list_temp in list_combined:
#            file_writer.writerow(list_temp)
#    return

# def fill_header_line(header_line):
#     current = ''
#     for i in range(0, len(header_line)):
#         if len(header_line[i]) > 0:
#             current = header_line[i]
#         if len(header_line[i]) == 0:
#             header_line[i] = current
#     return header_line
#
#
# def read_header_lines(file_csv):
#     f = open(file_csv, 'rb')
#     reader = csv.reader(f)
#     header_type = []
#     header_mod = []
#     header_time = []
#     for i in range(0, 3):
#         header_line = reader.next()
#         if header_line[0] == 'Type':
#             header_type = fill_header_line(header_line)
#         if header_line[0] == 'Mod':
#             header_mod = fill_header_line(header_line)
#         if header_line[0] == 'Time':
#             header_time = fill_header_line(header_line)
#     if len(header_type) == 0:
#         header_type = header_mod
#     if len(header_mod) == 0:
#         header_mod = header_type
#     f.close()
#     return header_type[1:], header_mod[1:], header_time[1:]
#
#
# HEADER_FIRST = ['Mod', 'Type', 'Time']
# def read_subject_lines(file_csv):
#     type, mod, time = read_header_lines(file_csv)
#     list_mod, time_points = np.unique(mod, return_counts=True)
#     f = open(file_csv, 'rb')
#     dict1={}
#     with open(file_csv, "rb") as infile:
#         reader = csv.reader(infile)
#         for row in reader:
#             if not row[0] in HEADER_FIRST:
#                 dict1[row[0]] = {key: [] for key in list_mod}
#                 for i in range(1, len(row)):
#                     dict1[row[0]][mod[i-1]].append(row[i])
#     f.close()
#     return dict1
#
# def read_database(file_csv, csv_fields=['subject', 'input_img',
#                                         'output_img', 'output_csv',
#                                         'additional_img','input_csv',
#                                         'additional_csv']):
#     f = open(file_csv, 'rb')
#     dict1 = {}
#
#     with open(file_csv, "rb") as infile:
#         reader = csv.reader(infile)
#         for row in reader:
#             dict1[row[0]] = {key: value for key,value
#                              in zip(csv_fields[1:], row[1:])}
#     f.close()
#     return dict1
