import utilities.constraints_classes as cc
import utilities.misc_csv as misc_csv
from utilities.subject import Subject

class CSVTable(object):
    """
    This class converts csv files into a nested list _csv_file
    a method is provided to output a list of Subject objects based on the
    _csv_file. The input can be a single csv file, with each cell
    corresponding to one 5-D image, or multiple csv files, with each cell
    corresponding to a single modality file. In the case of multiple csv
    files, the subject name is matched to have a joint _csv_file
    """
    def __init__(self, csv_file=None, csv_dict=None):
        self._csv_table = None
        if csv_file is not None:
            self.create_by_reading_single_csv(csv_file)
        if csv_dict is not None:
            self.create_by_join_multiple_csv_files(**csv_dict)
        if self._csv_table is None:
            raise RuntimeError('unable to read csv files into a nested list')


    def create_by_join_multiple_csv_files(self,
                                          input_image_file,
                                          target_image_file=None,
                                          weight_map_file=None,
                                          target_note_file=None,
                                          allow_missing=True):
        input_image_id, input_image_fullname = \
            misc_csv.create_array_files_from_csv(input_image_file,
                                                 allow_missing=allow_missing)
        # TODO: currently hard coded, to make it flexible in the future
        header = ('target_image_file', 'weight_map_file', 'target_note_file')
        csv_to_join = {header[0]: target_image_file,
                       header[1]: weight_map_file,
                       header[2]: target_note_file}

        # try to do pairwise matching between input_image_file and the others
        joint_id = None
        matches = {}
        for f in header:
            csv_file = csv_to_join[f]
            if csv_file is None:
                matches[f] = (None, None)
                continue
            # read single csv file (first column: id, rest column: image name)
            csv_id, matched_fullnames = misc_csv.create_array_files_from_csv(
                csv_file, allow_missing=allow_missing)
            # find matching between first column and 'input_image_file'
            joint_id, matched_index, _, _ = misc_csv.match_second_degree(
                input_image_id, csv_id)
            matches[f] = (matched_index, matched_fullnames)
        # no table join, using input_image_file as the final id of csv_table
        if joint_id is None:
            joint_id = misc_csv.remove_duplicated_names(input_image_id)
            joint_id = ['_'.join(sublist) for sublist in input_image_id]

        # create matching result to a joint csv table
        self._csv_table = []
        for (i, name) in enumerate(joint_id):
            # construct a row of the csv_table
            joint_csv_row = []
            joint_csv_row.append(name)
            joint_csv_row.append(input_image_fullname[i])
            # add other matched paths from other csv files
            for f in header:
                matched_index = matches[f][0]
                if matched_index is None:
                    joint_csv_row.append('')
                else:
                    matched_fullnames = matches[f][1]
                    joint_csv_row.append(matched_fullnames[matched_index[i]])
            self._csv_table.append(joint_csv_row)

    def create_by_reading_single_csv(self, csv_file):
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

    def to_subject_list(self):
        subject_list = []
        # interp_order = self.guess_interp_from_loss()
        interp_order_fields = cc.InputList([3], [0], [3], None, None)
        for row in self._csv_table:
            input_files = cc.CSVCell(row[1])
            output_files = cc.CSVCell(row[2]) if row[2] != '' else None
            weight_files = cc.CSVCell(row[3]) if row[3] != '' else None
            input_txt_files = cc.CSVCell(row[4]) if row[4] != '' else None
            file_path_list = cc.InputList(input_files,
                                          output_files,
                                          weight_files,
                                          input_txt_files,
                                          None)
            new_subject = Subject(row[0], file_path_list, interp_order_fields)
            subject_list.append(new_subject)
        return subject_list


        # def guess_interp_from_loss(self):
        #    categorical = ['cross_entropy', 'dice']
        #    interp_order = []
        #    for l in self.loss:
        #        order = 0 if l in categorical else 3
        #        interp_order.append(order)
        #    return interp_order
