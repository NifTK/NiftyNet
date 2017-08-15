from __future__ import absolute_import, print_function

import os

import tensorflow as tf

import nibabel as nib

import numpy as np

from niftynet.utilities.subject import MultiModalFileList
from niftynet.utilities.subject import Subject


class OrientationTest(tf.test.TestCase):

    def test_reorientation(self):

        test_name = 'dummy_LPS'
        t1_path = os.path.join('../data/dummies', '%s.nii.gz' %test_name)
        resaved_path = '../data/dummies'
        input = MultiModalFileList([[t1_path]])

        a_subject = Subject(name=test_name)
        a_subject.set_all_columns(input, None, None, None)
        out_1 = a_subject.load_column((0), do_reorientation=True,
                                      do_resampling=False)
        data_test = nib.load(
            '../data/dummies/dummy_RAS.nii.gz').get_data()
        assert np.sum(np.squeeze(out_1['input_image_file'].data) - data_test) == 0
        a_subject.save_network_output(out_1['input_image_file'].data,
                                      '../data/dummies/')
        data_origin = nib.load(t1_path)
        data_resaved = nib.load(os.path.join(resaved_path,'%s_niftynet_out.nii'
                                %test_name))
        print('reloaded')
        assert np.sum(data_origin.affine - data_resaved.affine) == 0
        assert np.sum(data_origin.get_data() - np.squeeze(
            data_resaved.get_data())) == 0
        print(a_subject._read_original_affine())
        print(a_subject._read_original_pixdim())
        print(out_1.keys())


if __name__ == "__main__":
    tf.test.main()
