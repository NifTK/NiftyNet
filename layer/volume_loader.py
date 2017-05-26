from random import shuffle
from .base import Layer

class VolumeLoaderLayer(Layer):
    """
    This class manages the loading step, i.e., return subject's data
    by searching user provided path and modality constraints.
    The volumes are resampled/reoriented if required.

    This class maintains a list of subjects, where each element of the list
    is a Patient object.
    """

    def __init__(self,
                 csv_reader,
                 standardisor,
                 do_shuffle=False,
                 name='volume_loader'):

        super(VolumeLoaderLayer, self).__init__(name=name)

        self.csv_table = csv_reader
        self.standardisor = standardisor
        self.do_shuffle = do_shuffle

        self.subject_list = None
        self.current_id = None
        self.__initialise_subject_list()

    def __initialise_subject_list(self):
        """
        provide a list of subjects, the subjects are constructed from csv_table
        data. These are used to train a histogram normalisation reference.
        """
        self.subject_list = self.csv_table.to_subject_list()
        if self.do_shuffle:
            shuffle(self.subject_list)
        self.current_id = -1


    def layer_op(self,
                 do_reorientation=False,
                 do_resampling=False,
                 do_normalisation=False,
                 do_whitening=False,
                 interp_order=(3, 0, 3)):
        """
        Call this function to get the next subject's image data.
        """
        # go to the next subject in the list (avoid running out of the list)
        if self.do_shuffle:
            self.current_id = np.random.randint(0, len(self.patients))
        else:
            self.current_id = self.current_id + 1
        current_subject = self.subject_list[self.current_id]
        #print current_subject
        subject_dict = current_subject.load_columns((0, 1, 2),
                                                    do_reorientation,
                                                    do_resampling,
                                                    interp_order)

        image = subject_dict['input_image_file']
        label = subject_dict['target_image_file']
        weight = subject_dict['weight_map_file']

        if not self.standardisor.is_ready(do_normalisation, do_whitening):
            self.standardisor.train_normalisation_ref(self.subject_list)
        image = self.standardisor(image, do_normalisation, do_whitening)
        return image, label, weight, self.current_id

    @property
    def has_next(self):
        if self.do_shuffle:
            return True
        if self.current_id < len(self.subject_list) - 1:
            return True
        return False

    def num_modality(self, column_id):
        return self.subject_list[0].column(column_id).num_modality


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
