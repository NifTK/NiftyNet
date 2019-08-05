# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from niftynet.io.image_type import ImageFactory
from niftynet.io.image_type import SpatialImage2D
from niftynet.io.image_type import SpatialImage3D
from niftynet.io.image_type import SpatialImage4D
from niftynet.io.image_type import SpatialImage5D
from niftynet.io.misc_io import set_logger
from tests.niftynet_testcase import NiftyNetTestCase

CASE_2D = 'testing_data/2d_3_000044.nii.gz'
CASE_3D_a = 'testing_data/1040_o_T1_time_01.nii.gz'
CASE_3D_b = 'testing_data/1023_o_T1_time_01.nii.gz'
CASE_5D = 'testing_data/pat2__niftynet_out.nii.gz'


class ImageTypeTest(NiftyNetTestCase):
    def test_2d(self):
        image = ImageFactory.create_instance(
            file_path=CASE_2D,
            name='2d_image',
            interp_order=3,
            output_pixdim=None,
            output_axcodes=None,
            loader=None)
        self.assertIsInstance(image, SpatialImage2D)
        output = image.get_data()
        self.assertEqual(np.dtype(np.float32), image.dtype[0])
        self.assertEqual(2, image.spatial_rank)
        self.assertAllClose(np.array([128, 128, 1, 1, 1]), output.shape)
        self.assertAllClose(np.array([128, 128, 1, 1, 1]), image.shape)

    def test_3d(self):
        image = ImageFactory.create_instance(
            file_path=CASE_3D_a,
            name='3d_image',
            interp_order=3,
            output_pixdim=None,
            output_axcodes=None,
            loader=None)
        self.assertIsInstance(image, SpatialImage3D)
        self.assertEqual(3, image.spatial_rank)
        output = image.get_data()
        self.assertAllClose(np.array([256, 168, 256, 1, 1]), output.shape)
        self.assertAllClose(np.array([256, 168, 256, 1, 1]), image.shape)

    def test_3d_reorientation(self):
        image = ImageFactory.create_instance(
            file_path=CASE_3D_a,
            name='3d_image',
            interp_order=3,
            output_pixdim=None,
            output_axcodes='ALS',
            loader=None)
        self.assertIsInstance(image, SpatialImage3D)
        self.assertEqual(3, image.spatial_rank)
        output = image.get_data()
        self.assertAllClose(np.array([168, 256, 256, 1, 1]), output.shape)
        self.assertAllClose(np.array([168, 256, 256, 1, 1]), image.shape)

    def test_3d_resample(self):
        image = ImageFactory.create_instance(
            file_path=CASE_3D_a,
            name='3d_image',
            interp_order=3,
            output_pixdim=(3.5, 0.9, 8),
            output_axcodes=(None,),
            loader=None)
        self.assertIsInstance(image, SpatialImage3D)
        self.assertEqual(3, image.spatial_rank)
        output = image.get_data()
        self.assertAllClose(np.array([71, 278, 31, 1, 1]), output.shape)
        self.assertAllClose(np.array([71, 278, 31, 1, 1]), image.shape)

    def test_3d_resample_reorientation(self):
        image = ImageFactory.create_instance(
            file_path=CASE_3D_a,
            name='3d_image',
            interp_order=3,
            output_pixdim=((7, 1.2, 8),),
            output_axcodes=('ASL',),
            loader=None)
        self.assertIsInstance(image, SpatialImage3D)
        output = image.get_data()
        self.assertAllClose(np.array([36, 208, 31, 1, 1]), output.shape)
        self.assertAllClose(np.array([36, 208, 31, 1, 1]), image.shape)

    def test_multiple_3d_as_4d(self):
        image = ImageFactory.create_instance(
            file_path=(CASE_3D_a, CASE_3D_b),
            name=('3d_image_a', '3d_image_b'),
            interp_order=(3, 3),
            output_pixdim=None,
            output_axcodes=(None, None),
            loader=(None, None))
        self.assertIsInstance(image, SpatialImage4D)
        self.assertEqual(3, image.spatial_rank)
        output = image.get_data()
        self.assertAllClose(np.array([256, 168, 256, 1, 2]), output.shape)
        self.assertAllClose(np.array([256, 168, 256, 1, 2]), image.shape)

    def test_multiple_3d_as_4d_resample(self):
        with self.assertRaisesRegexp(ValueError, 'concatenation'):
            image = ImageFactory.create_instance(
                file_path=(CASE_3D_a, CASE_3D_b),
                name=('3d_image_a', '3d_image_b'),
                interp_order=(3, 3),
                output_pixdim=((5, 4, 10.0), (2, 4, 1.0)),
                output_axcodes=(None, None),
                loader=(None, None))
            _ = image.get_data()
        image = ImageFactory.create_instance(
            file_path=(CASE_3D_a, CASE_3D_b),
            name=('3d_image_a', '3d_image_b'),
            interp_order=(3, 3),
            output_pixdim=(2.6, 0.9, 5),
            output_axcodes=(None, None),
            loader=(None, None))
        self.assertIsInstance(image, SpatialImage4D)
        self.assertEqual(3, image.spatial_rank)
        output = image.get_data()
        self.assertAllClose(np.array([96, 278, 50, 1, 2]), output.shape)
        self.assertAllClose(np.array([96, 278, 50, 1, 2]), image.shape)

    def test_multiple_3d_as_4d_reorientation(self):
        with self.assertRaisesRegexp(ValueError, 'concatenation'):
            image = ImageFactory.create_instance(
                file_path=(CASE_3D_a, CASE_3D_b),
                name=('3d_image_a', '3d_image_b'),
                interp_order=(0, 3),
                output_pixdim=(None, None),
                output_axcodes=(('A', 'S', 'L'), ('L', 'S', 'A')),
                loader=(None, None))
            _ = image.get_data()
        image = ImageFactory.create_instance(
            file_path=(CASE_3D_a, CASE_3D_b),
            name=('3d_image_a', '3d_image_b'),
            interp_order=(3, 0),
            output_pixdim=(None, None),
            output_axcodes=(('A', 'S', 'L'), ('A', 'S', 'L')),
            loader=(None, None))
        self.assertIsInstance(image, SpatialImage4D)
        output = image.get_data()
        self.assertAllClose(np.array([168, 256, 256, 1, 2]), output.shape)
        self.assertAllClose(np.array([168, 256, 256, 1, 2]), image.shape)

    def test_multiple_3d_as_4d_resample_reorientation(self):
        with self.assertRaisesRegexp(ValueError, 'concatenation'):
            image = ImageFactory.create_instance(
                file_path=(CASE_3D_a, CASE_3D_b),
                name=('3d_image_a', '3d_image_b'),
                interp_order=(3, 0),
                output_pixdim=((8.0, 6.0, 2.0), (7.0, 6.0, 10.0)),
                output_axcodes=(('A', 'S', 'L'), ('L', 'S', 'A')),
                loader=(None, None))
            _ = image.get_data()
        image = ImageFactory.create_instance(
            file_path=(CASE_3D_a, CASE_3D_b),
            name=('3d_image_a', '3d_image_b'),
            interp_order=(3, 3),
            output_pixdim=((5, 2.0, 6), (5, 2.0, 6)),
            output_axcodes='ASL',
            loader=(None, None))
        self.assertIsInstance(image, SpatialImage4D)
        output = image.get_data()
        self.assertAllClose(np.array([50, 125, 42, 1, 2]), output.shape)
        self.assertAllClose(np.array([50, 125, 42, 1, 2]), image.shape)

    def test_5d(self):
        image = ImageFactory.create_instance(
            file_path=CASE_5D,
            name='5d_image',
            interp_order=3,
            output_pixdim=None,
            output_axcodes=None,
            loader=None)
        self.assertIsInstance(image, SpatialImage5D)
        self.assertEqual(3, image.spatial_rank)
        output = image.get_data()
        self.assertAllClose(np.array([208, 256, 256, 1, 1]), output.shape)
        self.assertAllClose(np.array([208, 256, 256, 1, 1]), image.shape)

    def test_5d_resample(self):
        image = ImageFactory.create_instance(
            file_path=CASE_5D,
            name='5d_image',
            interp_order=1,
            output_pixdim=(4, 8, 6.0),
            output_axcodes=None,
            loader=None)
        self.assertIsInstance(image, SpatialImage5D)
        output = image.get_data()
        self.assertAllClose(np.array([58, 35, 49, 1, 1]), output.shape)
        self.assertAllClose(np.array([58, 35, 49, 1, 1]), image.shape)

    def test_5d_reorientation(self):
        image = ImageFactory.create_instance(
            file_path=(CASE_5D,),
            name='5d_image',
            interp_order=3,
            output_pixdim=(None,),
            output_axcodes=(('S', 'A', 'L'),),
            loader=(None,))
        self.assertIsInstance(image, SpatialImage5D)
        output = image.get_data()
        self.assertAllClose(np.array([256, 256, 208, 1, 1]), output.shape)
        self.assertAllClose(np.array([256, 256, 208, 1, 1]), image.shape)

    def test_5d_reorientation_resample(self):
        image = ImageFactory.create_instance(
            file_path=(CASE_5D,),
            name='5d_image',
            interp_order=3,
            output_pixdim=((8, 9, 10),),
            output_axcodes=('RSA',),
            loader=(None,))
        self.assertIsInstance(image, SpatialImage5D)
        output = image.get_data()
        self.assertAllClose(np.array([29, 33, 28, 1, 1]), output.shape)
        self.assertAllClose(np.array([29, 33, 28, 1, 1]), image.shape)


if __name__ == "__main__":
    set_logger()
    tf.test.main()
