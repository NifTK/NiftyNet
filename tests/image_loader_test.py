# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import numpy as np
import tensorflow as tf
import niftynet.io.image_loader as image_loader
from tests.niftynet_testcase import NiftyNetTestCase

CASE_NIBABEL_3D = 'testing_data/FLAIR_1023.nii.gz'
CASE_LOGO_2D = 'niftynet-logo.png'

class ImageLoaderTest(NiftyNetTestCase):
    def test_nibabel_3d(self):
        data = image_loader.load_image_obj(CASE_NIBABEL_3D).get_data()
        self.assertAllClose(data.shape, (256, 168, 256))

    def load_2d_image(self, loader=None):
        data = image_loader.load_image_obj(CASE_LOGO_2D, loader=loader).get_data()
        self.assertAllClose(data.shape, (400, 677, 1, 1, 4))

    def test_convert_bool(self):
        boolarr=np.ones((256,256,256),np.bool)
        img=image_loader.image2nibabel(boolarr)

    def test_2d_loaders(self):
        with self.assertRaisesRegexp(ValueError, ''):
            self.load_2d_image('test')
        self.load_2d_image()
        for _loader in image_loader.AVAILABLE_LOADERS.keys():
            print('testing {}'.format(_loader))
            if _loader == 'nibabel':
                continue # skip nibabel for 2d image
            if _loader == 'dummy':
                continue # skip the toy example
            self.load_2d_image(_loader)

    def test_all_data(self):
        folder = 'testing_data'
        all_files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))]

        for f in all_files:
            if f.endswith('nii.gz'):
                loaded_shape = image_loader.load_image_obj(f).get_data().shape
                print(loaded_shape)
                self.assertGreaterEqual(5, len(loaded_shape))
            else:
                with self.assertRaisesRegexp(ValueError, ''):
                    image_loader.load_image_obj(f)


if __name__ == "__main__":
    tf.test.main()
