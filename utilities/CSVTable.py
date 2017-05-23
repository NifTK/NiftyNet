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
        input_image_id, input_fullname =\
            misc_csv.create_array_files_from_csv(input_image_file, allow_missing)
        target_image_id, target_fullname = \
            misc_csv.create_array_files_from_csv(target_image_file, allow_missing)
        weight_map_id, weight_map_fullname = \
            misc_csv.create_array_files_from_csv(weight_map_file, allow_missing)
        target_note_id, target_note_fullname = \
            misc_csv.create_array_files_from_csv(target_note_file, allow_missing)
