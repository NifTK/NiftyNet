import utilities.misc_csv as misc_csv

class CSVTable(object):

    def __init__(self,
                 input_image,
                 target_image,
                 weight_map,
                 target_note):

        self.input_image = input_image
        self.target_image = target_image
        self.weight_map = weight_map
        self.target_note = target_note

        self.input_interp_order = 3
        self.target_interp_order = 0

    def create_from_separate_csv_files(self,
                                       input_image_file,
                                       target_image_file=None,
                                       weight_map_file=None,
                                       target_note_file=None,
                                       allow_missing=True):
        input_image_id, input_image_fullname =\
            misc_csv.create_array_files_from_csv(input_image_file,
                                                 allow_missing)
        if target_image_file is not None:
            target_image_id, target_image_fullname = \
                misc_csv.create_array_files_from_csv(target_image_file,
                                                     allow_missing)
            input_target_id, input_target_index, _, _ = \
                misc_csv.match_second_degree(input_image_id, target_image_id)

        if weight_map_file is not None:
            weight_map_id, weight_map_fullname, _, _ = \
                misc_csv.create_array_files_from_csv(weight_map_file,
                                                     allow_missing)
            input_weight_map_id, input_weight_map_index = \
                misc_csv.match_second_degree(input_image_id, weight_map_id)
        if target_note_file is not None:
            target_note_id, target_note_fullname, _, _ = \
                misc_csv.create_array_files_from_csv(target_note_file,
                                                     allow_missing)
            input_note_id, input_note_index, _, _ = \
                misc_csv.match_second_degree(input_image_id, weight_map_id)

