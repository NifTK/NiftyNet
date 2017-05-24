import tensorflow as tf
from utilities.subject import Subject
import utilities.constraints_classes as cc

class SubjectTest(tf.test.TestCase):

    def test_new_subject(self):

        test_name = '1023'
        t1_path = 'testing_data/1023_o_T1_time_01.nii.gz'
        label_path = 'testing_data/T1_1023_NeuroMorph_Parcellation.nii.gz'
        input = cc.CSVCell([[t1_path]])
        output = cc.CSVCell([[label_path]])

        a_subject = Subject(name=test_name)
        a_subject.set_all_columns(input, output, None, None)

        print(a_subject._read_original_affine())
        print(a_subject._read_original_pixdim())

        out = a_subject.load_columns((0, 1, 2),
                                     do_reorient=False,
                                     do_resample=False)
        print out[0].shape

        out_1 = a_subject.load_columns((0, 1, 2),
                                       do_reorient=True,
                                       do_resample=False)
        print out_1[0].shape

        out_2 = a_subject.load_columns((0, 1, 2),
                                       do_reorient=True,
                                       do_resample=True)
        print out_2[0].shape
        print(a_subject)

if __name__ == "__main__":
    tf.test.main()
