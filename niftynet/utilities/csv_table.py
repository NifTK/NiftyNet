# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import niftynet.utilities.misc_csv as misc_csv
from niftynet.utilities.subject import Subject


class SubjectTable(object):
    """
    Generic class for holding a subject list
    """

    def __init__(self, subject_table, modality_names=None):
        self._subject_table = subject_table
        self.modality_names = modality_names

    def to_subject_list(self):
        return [Subject.from_csv_row(row, self.modality_names)
                for row in self._subject_table]


class CSVTable(SubjectTable):
    """
    This class converts csv files into a nested list _csv_file
    a method is provided to output a list of Subject objects based on the
    _csv_file. The input can be a single csv file, with each cell
    corresponding to one 5-D image, or multiple csv files, with each cell
    corresponding to a single modality file. In the case of multiple csv
    files, the subject name is matched to have a joint _csv_file
    """

    def __init__(self,
                 csv_file=None,
                 csv_dict=None,
                 modality_names=None,
                 allow_missing=True):
        self.allow_missing = allow_missing
        self._csv_table = None
        if csv_file is not None:
            self.create_by_reading_single_csv(csv_file)
        if csv_dict is not None:
            self.create_by_joining_multiple_csv_files(**csv_dict)
        if self._csv_table is None:
            raise RuntimeError('unable to read csv files into a nested list')
        super(CSVTable, self).__init__(subject_table=self._csv_table,
                                       modality_names=modality_names)

    def create_by_joining_multiple_csv_files(self, **csv_dict):
        """
        This function creates a CSV table from multiple files read from the
        argument csv_dict. This is used when the list of inputs, targets,
        weights... are listed in separate csv files
        """


        header = Subject.fields
        csv_to_join = {}
        for h in header:
            try:
                csv_to_join[h] = csv_dict[h]
            except KeyError:
                print("The csv_dict input should have the following keys:\n"
                      + "{}\n".format(header)
                      + "Each value should be a filename of csv file\n"
                      + "where each csv file contains at least two columns\n"
                      + "  - the first column: subject id\n"
                      + "  - the rest columns: image filename\n"
                      + "subject_id will be used to join multiple csv files.")

        input_image_id, input_image_fullname = \
            misc_csv.load_subject_and_filenames_from_csv_file(
                csv_to_join[header[0]], self.allow_missing, None)
        # try to do pairwise matching between input_image_file and the others
        joint_id = None
        matches = {}
        for f in header[1:]:
            csv_file = csv_to_join[f]
            if csv_file is None:
                matches[f] = (None, None)
                continue
            # read single csv file (first column: id, rest column: image name)
            csv_id, matched_fullnames = \
                misc_csv.load_subject_and_filenames_from_csv_file(
                    csv_file, self.allow_missing, None)
            # find matching between first column and 'input_image_file'
            joint_id, matched_index, _, _ = misc_csv.match_second_degree(
                input_image_id, csv_id)
            matches[f] = (matched_index, matched_fullnames)
        # no table join, using input_image_file as the final id of csv_table
        if joint_id is None:
            joint_id = misc_csv.remove_duplicated_names(input_image_id)
            joint_id = ['_'.join(sublist) for sublist in joint_id]

        # create matching result to a joint csv table
        self._csv_table = []
        for (i, name) in enumerate(joint_id):
            # construct a row of the csv_table
            joint_csv_row = [name, input_image_fullname[i]]
            # add other matched paths from other csv files
            for f in header[1:]:
                matched_index = matches[f][0]
                if matched_index is None:
                    joint_csv_row.append('')
                else:
                    matched_fullnames = matches[f][1]
                    joint_csv_row.append(matched_fullnames[matched_index[i]])
            self._csv_table.append(joint_csv_row)

    def create_by_reading_single_csv(self, csv_file):
        '''
        Creates the csv table by reading a single csv_file that contains for
        each field a single file input.
        :param csv_file:
        :return:
        '''
        self._csv_table = []
        with open(csv_file, "rb") as infile:
            reader = csv.reader(infile)
            csv_row = []
            for row in reader:
                csv_row.append(row[0])
                csv_row.append([[row[1]]])
                csv_row.append([[row[2]]])
                csv_row.append([[row[3]]])
                csv_row.append([[row[4]]])
                self._csv_table.append(csv_row)

