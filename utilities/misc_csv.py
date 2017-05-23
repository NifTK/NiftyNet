import csv
import os
import warnings
from difflib import SequenceMatcher

import constraints_classes as cc

HEADER_FIRST = ['Mod', 'Type', 'Time']


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


# From a unique csv file with for each subject the list of files to use,
# build the 2d array of files to load for each subject and create the overall
#  list of such arrays. num_modality indicates the
# number of modalities before going to further time point
def create_array_files_from_csv(csv_file, numb_mod=None, allow_missing=True):
    if csv_file is None:
        return [], []
    list_subjects = []
    list_filenames = []
    with open(csv_file, "rb") as infile:
        reader = csv.reader(infile)
        for row in reader:
            if ('' in row) and (not allow_missing):
                continue
            subject_name, list_files = [row[0]], row[1:]
            list_subjects.append(subject_name)
            numb_mod = len(list_files) if numb_mod is None else numb_mod
            grouped_time_points = [list_files[i:(i+numb_mod)]
                                   for i in range(0, len(list_files), numb_mod)]
            list_filenames.append(grouped_time_points)
    return list_subjects, list_filenames


# Create the list of files arrays for each subject according to csv format
# input (either a unique csv file with already build 5d data or multiple csv
# files according to data field (input, output, weight input_txt or output_txt)
def create_array_files(csv_file=None, csv_list=None, number=None):
    if csv_list is not None:
        if (csv_list.input is None) or (not os.path.exists(csv_list.input)):
            warnings.warn("No input csv, no array list produced")
            return None
        _, list_filenames = create_array_files_from_csv(csv_list.input, number)
        return list_filenames
    if csv_file is not None:
        with open(csv_file, "rb") as infile:
            reader = csv.reader(infile)
            list_filenames = []
            for row in reader:
                # TODO: check, maybe row[1:]
                list_filenames.append([[row[1]]])
        return list_filenames
    warnings.warn("No csv file available, no array list produced")
    return None


# try to find a direct match between two arrays of list of possible names.
def match_first_degree(name_list1, name_list2):
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
        for name in name_list1[i]:
            if name in flatten_list1:
                init_match2[i] = name
                ind_match2[i] = indflat_1[flatten_list1.index(name)]
                break
    return init_match1, init_match2, ind_match1, ind_match2


# Find the maximum overlap of n in the list of strings list_names. Returns
# the matched sequence and the corresponding index of the corresponding
# element in the list
def find_max_overlap_in_list(name, list_names):
    match_max = 0
    match_seq = ''
    match_orig = ''
    for test in list_names:
        match = SequenceMatcher(None, name, test).find_longest_match(
                0, len(name), 0, len(test))
        if match.size > match_max:
            match_max = match.size
            match_seq = test[match.b:(match.b + match.size)]
            match_orig = test
    return match_seq, list_names.index(match_orig)


# Perform the double matching between two lists of list of possible names.
# First find the direct matches, remove them from the ones still to match and
#  match the remainding ones using the maximum overlap. Returns the name
# match for each list, and the index correspondences.
def match_second_degree(name_list1, name_list2):
    if name_list1 is None or name_list2 is None:
        return None, None, None, None
    init_match1, init_match2, ind_match1, ind_match2 = match_first_degree(
        name_list1, name_list2)
    reduced_list1 = [names for names in name_list1
        if init_match1[name_list1.index(names)] == '']
    reduced_list2 = [names for names in name_list1
        if init_match1[name_list1.index(names)] == '']
    redflat_1 = [item for sublist in reduced_list1 for item in sublist]
    indflat_1 = [i for i in range(0, len(init_match1)) for item in
                 name_list1[i] if init_match1[i] == '']
    redflat_2 = [item for sublist in reduced_list2 for item in sublist]
    indflat_2 = [i for i in range(0, len(init_match2)) for item in
                 name_list2[i] if init_match2[i] == '']
    for i in range(0, len(name_list1)):
        if init_match1[i] == '':
            for n in name_list1[i]:
                init_match1[i], index = find_max_overlap_in_list(n, redflat_2)
                ind_match1[i] = indflat_1[index]
    for i in range(0, len(name_list2)):
        if init_match2[i] == '':
            for n in name_list1[i]:
                init_match2[i], index = find_max_overlap_in_list(n, redflat_1)
                ind_match2[i] = indflat_2[index]
    return init_match1, ind_match1, init_match2, ind_match2


# From a list of list of names and a list of list of files that are
# associated, find the name correspondance and therefore the files associations
def combine_list_constraint_for5d(name_list, list_files):
    name_length = [len(names) for names in name_list]
    ind_max = name_length.index(max(name_length))
    name_tot = []
    ind_tot = []
    name_max = name_list[ind_max]
    for c in range(0, len(list_files)):
        name_match, ind_match, _, _ = match_second_degree(name_max,
                                                          name_list[c])
        name_max = name_match if c == ind_max else name_max
        name_tot.append(name_match)
        ind_tot.append(ind_match)

    list_combined = []
    for (i, name) in enumerate(name_max):
        list_temp = []
        list_temp.append(name)
        for c in range(0, len(list_files)):
            output = list_files[c][ind_tot[c][i]]
            list_temp.append(output)
        list_combined.append(list_temp)
    return list_combined


def combine_list_constraint(name_list, list_files):
    name_match_io, ind_io, _, _ = match_second_degree(name_list.input,
                                                      name_list.output)
    name_match_iw, ind_iw, _, _ = match_second_degree(name_list.input,
                                                      name_list.weight)
    name_match_iit, ind_iit, _, _ = match_second_degree(name_list.input,
                                                        name_list.input_txt)
    name_match_iot, ind_iot, _, _ = match_second_degree(name_list.input,
                                                        name_list.output_txt)
    if name_list.input is None:
        raise ValueError("There is no input! Please do check your constraints")

    if not name_list.output is None:
        list_to_use = name_match_io
    elif not name_list.weight is None:
        list_to_use = name_match_iw
    elif not name_list.input_txt is None:
        list_to_use = name_match_iit
    elif not name_list.output_txt is None:
        list_to_use = name_match_iot
    else:
        warnings.warn("You have only an input...")
        list_temp = remove_duplicated_names(name_list.input)
        list_to_use = ['_'.join(sublist) for sublist in list_temp]

    list_compare = []
    for (i, name) in enumerate(list_to_use):
        input = list_files.input[i]
        output = list_files.output[ind_io[i]] if ind_io is not None else ''
        weight = list_files.weight[ind_iw[i]] if ind_iw is not None else ''
        input_txt = list_files.weight[ind_iit[i]] if ind_iit is not None else ''
        output_txt = list_files.output_txt[ind_iot[i]] if ind_iot is not None else ''
        list_temp = [name, input, output, weight, input_txt, output_txt]
        list_compare.append(list_temp)
    return list_compare



def remove_duplicated_names(name_list):
    flattened_list = [item for sublist in name_list for item in sublist]
    list_duplicated = [item for item in flattened_list
                       if flattened_list.count(item) > 1]
    duplicates_removed = []
    for names in name_list:
        duplicates_removed.append([name for name in names
                                   if name not in list_duplicated])
    return duplicates_removed


def create_csv_prepare5d(list_constraints, csv_file):
    name_tot = []
    list_tot = []
    for c in list_constraints:
        list_files, name_list = \
            cc.ConstraintSearch.create_list_from_constraint(c)
        name_list = remove_duplicated_names(name_list)
        name_tot.append(name_list)
        list_tot.append(list_files)
    list_combined = combine_list_constraint_for5d(name_tot, list_tot)
    with open(csv_file, 'wb') as csvfile:
        file_writer = csv.writer(csvfile, delimiter=',')
        for list_temp in list_combined:
            file_writer.writerow(list_temp)
    return


def create_csv(constraint_list, csv_file):
    list_input = None
    list_output = None
    list_weight = None
    list_input_txt = None
    list_output_txt = None
    name_output_txt = None
    name_input_txt = None
    name_input = None
    name_output = None
    name_weight = None
    if constraint_list.input is not None:
        list_input, name_input = \
            cc.ConstraintSearch.create_list_from_constraint(
                constraint_list.input)
        name_input = remove_duplicated_names(name_input)
    if constraint_list.output is not None:
        list_output, name_output = \
            cc.ConstraintSearch.create_list_from_constraint(
                constraint_list.output)
        name_output = remove_duplicated_names(name_output)
    if constraint_list.weight is not None:
        list_weight, name_weight = \
            cc.ConstraintSearch.create_list_from_constraint(
                constraint_list.weight)
        name_weight = remove_duplicated_names(name_weight)
    if constraint_list.input_txt is not None:
        list_input_txt, name_input_txt = \
            cc.ConstraintSearch.create_list_from_constraint(
                constraint_list.input_txt)
        name_input_txt = remove_duplicated_names(name_input_txt)
    if constraint_list.output_txt is not None:
        list_output_txt, name_output_txt = \
            cc.ConstraintSearch.create_list_from_constraint(
                constraint_list.output_txt)
        name_output_txt = remove_duplicated_names(name_output_txt)

    list_files_init = cc.InputList(list_input, list_output, list_weight,
                                   list_input_txt, list_output_txt)

    list_names_init = cc.InputList(name_input, name_output, name_weight,
                                   name_input_txt, name_output_txt)

    list_combined = combine_list_constraint(list_names_init, list_files_init)
    with open(csv_file, 'wb') as csvfile:
        file_writer = csv.writer(csvfile, delimiter=',')
        for list_temp in list_combined:
            file_writer.writerow(list_temp)
    return
