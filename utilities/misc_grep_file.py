import os
import misc_io as util

# Given a list of constraints to look for in a list of directories, creates
# the list of subjects that satisfy the conditions
def combine_list_subjects_compulsory(list_data_dir, list_constraint):
    if len(list_constraint) == 1:
        name_init = list_subjects(list_data_dir, list_constraint[0])
    else:
        name_init = list_subjects(list_data_dir, list_constraint[0])
        for c in range(1, len(list_constraint)):
            flag_use_possible = False
            if list_constraint[c] in MOD_LIST:
                flag_use_possible = True
            name_temp = list_subjects(list_data_dir,
                                      list_constraint[c], flag_use_possible)
            for n in name_init:
                if n not in name_temp:
                    name_init.remove(n)
    for subject in name_init:
        dict_mod = create_list_modalities_available(list_data_dir, subject)
        for c in list_constraint:
            if c in MOD_LIST:
                if len(dict_mod[c]) == 0:
                    name_init.remove(subject)
            else:
                flag_ok = False
                for other in dict_mod['other']:
                    if c in other:
                        flag_ok = True
                if not flag_ok:
                    name_init.remove(subject)
    return name_init



MOD_LIST = ['FLAIR', 'T1', 'T2', 'T1c', 'LABEL', 'SWI', 'T2s', 'PD']
POSSIBLE_STRINGS = {'T1': ['T1', 't1', 'T1w', 't1w'],
                    'T2': ['T2', 't2', 'T2w', 't2w'],
                    'T2s': ['T2s', 't2s', 'T2star', 't2star'],
                    'FLAIR': ['FLAIR', 'Flair', 'flair'],
                    'PD': ['PD', 'pd', 'pdw', 'PDw'],
                    'T1c': ['T1Gad', 'T1c', 't1c', 't1ce', 'T1ce'],
                    'SWI': ['SWI', 'swi'],
                    'LABEL': ['Label', 'LABEL', 'label', 'Parcellation',
                              'parcellation', 'seg', 'segmentation',
                              'Segmentation']}
# Provides the list of subjects that satisfy a specific condition (modality)
def list_subjects(list_data_dir, element=None, flag_use_possible=True):
    name_possible = []
    for d in list_data_dir:
        for f in os.listdir(d):
            path, name, ext = util.split_filename(f)
            split = name.split('_')
            if element is None:
                for x in split:
                    name_possible.append(x)
            else:
                for s in range(0, len(split)):
                    if flag_use_possible:
                        if split[s] in POSSIBLE_STRINGS[element]:
                            for x in split:
                                if x is not split[s]:
                                    name_possible.append(x)
                            # name_possible.append(suffix)
                    else:
                        if split[s] == element:
                            for x in split:
                                if x is not split[s]:
                                    name_possible.append(x)
    name_possible = list(set(name_possible))
    return name_possible

# Given a list of constraints to look for in a list of directories, creates
# the list of subjects that satisfy the conditions
def combine_list_subjects_compulsory(list_data_dir, list_constraint):
    if len(list_constraint) == 1:
        name_init = list_subjects(list_data_dir, list_constraint[0])
    else:
        name_init = list_subjects(list_data_dir, list_constraint[0])
        for c in range(1, len(list_constraint)):
            flag_use_possible = False
            if list_constraint[c] in MOD_LIST:
                flag_use_possible = True
            name_temp = list_subjects(list_data_dir,
                                      list_constraint[c], flag_use_possible)
            for n in name_init:
                if n not in name_temp:
                    name_init.remove(n)
    for subject in name_init:
        dict_mod = create_list_modalities_available(list_data_dir, subject)
        for c in list_constraint:
            if c in MOD_LIST:
                if len(dict_mod[c]) == 0:
                    name_init.remove(subject)
            else:
                flag_ok = False
                for other in dict_mod['other']:
                    if c in other:
                        flag_ok = True
                if not flag_ok:
                    name_init.remove(subject)
    return name_init


LABEL_STRINGS = ['Label', 'LABEL', 'label', 'Parcellation', 'parcellation',
                 'seg', 'segmentation', 'Segmentation']
DICT_MODALITIES = {'T1':'T1', 't1':'T1', 'T1w':'T1', 't1w':'T1',
'T2':'T2', 't2': 'T2', 'T2w':'T2', 't2w':'T2',
'FLAIR':'FLAIR', 'Flair':'FLAIR', 'flair':'FLAIR',
'PD':'PD', 'pd':'PD', 'pdw':'PD', 'PDw':'PD',
'T2s':'T2s', 't2s':'T2s', 'T2star':'T2s', 't2star':'T2s',
'SWI':'SWI', 'swi':'SWI',
'T1Gad':'T1c', 'T1c':'T1c', 't1c':'T1c','t1ce':'T1c','T1ce':'T1c'
}
def create_list_modalities_available(list_data_dir, subject_id):
    list_files = list_nifti_subject(list_data_dir, subject_id)
    dict_modalities = create_init_dict_modalities()
    list_multiple_classif = [] # Handles the case where file is classified
    # under multiple modalities
    for filename in list_files:
        path, name, ext = util.split_filename(filename)
        split_name = name.split('_')
        flag_numb_classified = 0
        for s in split_name:
            if s in LABEL_STRINGS:
                dict_modalities['other'].append(filename)
                flag_numb_classified += 1
            if s in DICT_MODALITIES.keys():
                dict_modalities[DICT_MODALITIES[s]].append(filename)
        if flag_numb_classified == 0:
            dict_modalities['other'].append(filename)
            if flag_numb_classified > 1:
                list_multiple_classif.append(filename)
    dict_modalities = adapt_list_modalities_available(dict_modalities,
                                                              list_multiple_classif)
    return dict_modalities


# Returns list of nifti filenames for given subject according to list of
# directories
def list_nifti_subject(list_data_dir, subject_id):
    file_list = []
    for data_dir in list_data_dir:
        for file_name in os.listdir(data_dir):
            path, name, ext = util.split_filename(file_name)
            split_name = name.split('_')
            if subject_id in split_name and 'nii' in ext:
                file_list.append(os.path.join(data_dir, file_name))
    return file_list


def create_init_dict_modalities():
    dict_modalities = {'T1': [],
                       'T2': [],
                       'PD': [],
                       'FLAIR': [],
                       'T1c': [],
                       'T2s': [],
                       'SWI': [],
                       'other': []}
    return dict_modalities

def adapt_list_modalities_available(dict_available, list_double):
    for element in list_double:
        # First check if in other (then means it is a label or segmentation
        # and should not be considered as a modality
        list_possible_mod = list_mod(element)
        if element in dict_available['other']:
            for m in list_possible_mod:
                dict_available[m].remove(element)
    return dict_available