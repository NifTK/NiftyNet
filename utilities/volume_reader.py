from random import shuffle

from nn.preprocess import HistNormaliser_bis
from utilities.CSVTable import CSVTable


class VolumePreprocessor(object):
    """
    This class manages the loading step, i.e., return subject's data
    by searching user provided path and modality constraints.
    The volumes are resampled/reoriented if required.

    This class maintains a list of subjects, where each element of the list
    is a Patient object.
    """

    def __init__(self,
                 dict_normalisation=None,
                 dict_masking=None,
                 csv_file=None,
                 csv_dict=None,
                 do_reorientation=False,
                 do_resampling=False,
                 do_normalisation=True,
                 do_whitening=True,
                 allow_missing=True,
                 output_columns=(0, 1, 2, 3),
                 interp_order=(3, 0, 3)):

        self.do_reorientation = do_reorientation
        self.do_resampling = do_resampling
        self.do_normalisation = do_normalisation
        self.do_whitening = do_whitening

        self.dict_normalisation = dict_normalisation

        self.csv_table = CSVTable(csv_file, csv_dict, allow_missing)

        self.standardisor = HistNormaliser_bis(
            self.dict_normalisation.hist_ref_file,
            self.dict_normalisation.path_to_train,
            dict_masking,
            self.dict_normalisation.norm_type,
            self.dict_normalisation.cutoff,
            dict_masking.mask_type, '')

        self.subject_list = self.create_subject_list()
        self.current_id = -1

        self.output_columns = output_columns
        self.interp_order = interp_order

    def create_subject_list(self):
        """
        provide a list of subjects, the subjects are constructed from csv_table
        data. These are used to train a histogram normalisation reference.
        """
        subjects = self.csv_table.to_subject_list()
        if self.do_normalisation:
            self.standardisor.train_normalisation_ref(subjects)
        return subjects

    def next_subject(self, do_shuffle=True):
        """
        Call this function to get the next subject's image data.
        """
        # go to the next subject in the list (avoid running out of the list)
        self.current_id = (self.current_id + 1) % len(self.subject_list)
        if do_shuffle:
            shuffle(self.subject_list)
        current_subject = self.subject_list[self.current_id]
        print current_subject
        subject_dict = current_subject.load_columns(self.output_columns,
                                                    self.do_reorientation,
                                                    self.do_resampling,
                                                    self.interp_order)

        image = subject_dict['input_image_file']
        label = subject_dict['target_image_file']
        weight = subject_dict['weight_map_file']
        if self.do_normalisation:
            image = self.standardisor.normalise(image)
        if self.do_whitening:
            image = self.standardisor.whiten(image)

        return image, label, weight, self.current_id

        # def normalise_subject_data_and_save(self, subject):
        #    if self.flags.flag_standardise:
        #        data_dict = subject.read_all_modalities(self.flags.flag_reorient,
        #                                                self.flags.flag_resample)
        #        data_dict.input = np.nan_to_num(data_dict.input)
        #        mask_array = self.make_mask_array(data_dict.input)
        #        data_dict.input = self.standardisor.normalise_data_array(
        #            data_dict.input, mask_array)
        #        name_norm_save = io.create_new_filename(
        #            subject.name + '.nii.gz',
        #            new_path=self.dict_normalisation.path_to_save,
        #            new_prefix='Norm')
        #        # Put back the array with the nifti conventions.
        #        data_nifti_format = np.swapaxes(data_dict.input, 4, 3)
        #        io.save_img(data_nifti_format, subject.name, [], name_norm_save,
        #                    filename_ref=subject.file_path_list.input.filename_ref,
        #                    flag_orientation=self.flags.flag_reorient,
        #                    flag_isotropic=self.flags.flag_resample)
        #        # TODO: save norm
        #        #subject._set_data_path(name_norm_save)
