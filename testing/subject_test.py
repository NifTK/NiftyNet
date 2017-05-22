import tensorflow as tf
from utilities.subject import Subject
import utilities.constraints_classes as cc

class SubjectTest(tf.test.TestCase):

    def test_new_subject(self):

        test_name = '1023'
        t1_path = 'testing_data/1023_o_T1_time_01.nii.gz'
        label_path = 'testing_data/T1_1023_NeuroMorph_Parcellation.nii.gz'
        input = cc.InputFiles(t1_path, [[t1_path]])
        output = cc.InputFiles(label_path, [[label_path]])

        data_list = cc.InputList(input, output, None, None, None)
        interp_orders = cc.InputList([3], [0], None, None, None)

        a_subject = Subject(name=test_name,
                            file_path_list=data_list,
                            interp_order=interp_orders)
        print(a_subject._read_original_affine())
        print(a_subject._read_original_pixdim())

        out = a_subject.read_all_modalities(do_reorient=False,
                                            do_resample=False)
        print out.input.shape

        out_1 = a_subject.read_all_modalities(do_reorient=True,
                                              do_resample=False)
        print out_1.input.shape

        out_2 = a_subject.read_all_modalities(do_reorient=True,
                                              do_resample=True)
        print out_2.input.shape
        print(a_subject)

if __name__ == "__main__":
    tf.test.main()
