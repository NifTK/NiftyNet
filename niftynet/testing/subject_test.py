from __future__ import absolute_import, print_function

import os

import tensorflow as tf

from niftynet.utilities.subject import MultiModalFileList
from niftynet.utilities.subject import Subject


class SubjectTest(tf.test.TestCase):

    def test_new_subject(self):

        test_name = '1023'
        t1_path = os.path.join('testing_data','1023_o_T1_time_01.nii.gz')
        label_path = os.path.join('testing_data','T1_1023_NeuroMorph_Parcellation.nii.gz')
        input = MultiModalFileList([[t1_path]])
        output = MultiModalFileList([[label_path]])
        weights = MultiModalFileList([[label_path]])
        comments = MultiModalFileList([[label_path]])

        a_subject = Subject(name=test_name)
        a_subject.set_all_columns(input, None, None, None)
        print(a_subject._read_original_affine())
        print(a_subject._read_original_pixdim())
        #out = a_subject.load_columns((0, 1, 2, 3),
        #                             do_reorientation=False,
        #                             do_resampling=False)
        #print(out.keys())

        #a_subject.set_all_columns(input, output, None, comments)
        #print(a_subject._read_original_affine())
        #print(a_subject._read_original_pixdim())
        #out = a_subject.load_columns((0, 1, 2, 3),
        #                             do_reorientation=False,
        #                             do_resampling=True)

        #print(out.keys())
        #out_1 = a_subject.load_columns((0, 1, 3),
        #                               do_reorientation=True,
        #                               do_resampling=False)
        #print(out_1.keys())

        #a_subject.set_all_columns(input, None, weights, comments)
        #out_2 = a_subject.load_columns((0, 1, 2),
        #                               do_reorientation=True,
        #                               do_resampling=True)
        #print(out_2.keys())
        #print(a_subject)

        a_subject.set_all_columns(input, output, weights, comments)
        out_2 = a_subject.load_columns((0, 1, 2),
                                       do_reorientation=True,
                                       do_resampling=True)
        print(out_2.keys())

if __name__ == "__main__":
    tf.test.main()
