import tensorflow as tf
from utilities.subject import Subject

class SubjectTest(tf.test.TestCase):

    def test_new_subject(self):
        test_name = '1023'
        file_path_dict = {'T1': 'testing_data/1023_o_T1_time_01.nii.gz',
                          'LABEL': 'testing_data/T1_1023_NeuroMorph_Parcellation.nii.gz'}
        a_subject = Subject(name=test_name,
                            file_path_dict=file_path_dict,
                            list_nn=['LABEL'])
        print a_subject._read_original_affine()
        print a_subject._read_original_pixdim()

        out = a_subject.read_all_modalities(do_reorient=False,
                                            do_resample=False)

        out_1 = a_subject.read_all_modalities(do_reorient=True,
                                              do_resample=False)

        out_2 = a_subject.read_all_modalities(do_reorient=True,
                                              do_resample=True)
        print a_subject

if __name__ == "__main__":
    tf.test.main()
