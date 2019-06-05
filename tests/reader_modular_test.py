# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

from PIL import Image
import nibabel as nib
import numpy as np
import tensorflow as tf

from niftynet.io.image_reader import ImageReader
from tests.niftynet_testcase import NiftyNetTestCase

# from niftynet.io.image_sets_partitioner import ImageSetsPartitioner


SEG_THRESHOLD = 100


def generate_2d_images():
    # Test 2D Image Loader

    # Generate 10 fake 2d grayscale and color images of size 100x100
    img_path = os.path.join(os.path.dirname(__file__), '..', 'testing_data')
    img_path = os.path.realpath(os.path.join(img_path, 'images2d'))
    if not os.path.isdir(img_path):
        os.makedirs(img_path)

    # Generate fake testing data
    for i in range(10):
        img1 = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
        gray = np.random.randint(0, 255, size=(100, 100)).astype(np.uint8)
        mask = ((gray > SEG_THRESHOLD) * 255).astype(np.uint8)
        Image.fromarray(img1).save(os.path.join(img_path, 'img%d_u.png' % i))
        Image.fromarray(gray).save(os.path.join(img_path, 'img%d_g.png' % i))
        Image.fromarray(mask).save(os.path.join(img_path, 'img%d_m.png' % i))
    return


def generate_2d_1d_images():
    img_path = os.path.join(os.path.dirname(__file__), '..', 'testing_data')
    img_path = os.path.realpath(os.path.join(img_path, 'images_x_1_y'))
    if not os.path.isdir(img_path):
        os.makedirs(img_path)
    for idx in range(3):
        image_data = np.random.randint(0, 255, size=(100, 1, 100))
        image_data = image_data.astype(np.uint8)
        nib_image = nib.Nifti1Image(image_data, np.eye(4))
        nib.save(nib_image,
                 os.path.join(img_path, 'x_1_y_{}.nii.gz'.format(idx)))
    return


def generate_3d_1_1_d_images():
    img_path = os.path.join(os.path.dirname(__file__), '..', 'testing_data')
    img_path = os.path.realpath(os.path.join(img_path, 'images_x_y_z_1_1'))
    if not os.path.isdir(img_path):
        os.makedirs(img_path)
    for idx in range(3):
        image_data = np.random.randint(0, 255, size=(50, 24, 42, 1, 1))
        image_data = image_data.astype(np.uint8)
        nib_image = nib.Nifti1Image(image_data, np.eye(4))
        nib.save(nib_image,
                 os.path.join(img_path, 'x_y_z_1_1_{}.nii.gz'.format(idx)))
    return


generate_2d_images()
generate_2d_1d_images()
generate_3d_1_1_d_images()

IMAGE_PATH_2D = os.path.join('.', 'testing_data', 'images2d')
IMAGE_PATH_2D_1 = os.path.join('.', 'example_volumes', 'gan_test_data')
IMAGE_PATH_2D_2 = os.path.join('.', 'testing_data', 'images_x_1_y')
IMAGE_PATH_3D = os.path.join('.', 'testing_data')
IMAGE_PATH_3D_1 = os.path.join('.', 'testing_data', 'images_x_y_z_1_1')


class Read2DTest(NiftyNetTestCase):
    def default_property_asserts(self, reader):
        self.assertDictEqual(reader.spatial_ranks, {'mr': 2})
        self.assertDictEqual(reader.input_sources, {'mr': ('mr',)})
        self.assertDictEqual(reader.shapes, {'mr': (100, 100, 1, 1, 1)})
        self.assertDictEqual(reader.tf_dtypes, {'mr': tf.float32})
        self.assertEqual(reader.names, ('mr',))
        self.assertEqual(len(reader.output_list), 30)

    def renamed_property_asserts(self, reader):
        # test properties
        self.assertDictEqual(reader.spatial_ranks, {'ct': 2})
        self.assertDictEqual(reader.input_sources, {'ct': ('mr',)})
        self.assertDictEqual(reader.shapes, {'ct': (100, 100, 1, 1, 1)})
        self.assertDictEqual(reader.tf_dtypes, {'ct': tf.float32})
        self.assertEqual(reader.names, ('ct',))
        self.assertEqual(len(reader.output_list), 30)

    def test_simple(self):
        data_param = {'mr': {'path_to_search': IMAGE_PATH_2D}}
        reader = ImageReader().initialise(data_param)
        # test properties
        self.default_property_asserts(reader)
        idx, data, interp = reader()

        # test output
        self.assertTrue('mr' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'mr': (1,)})
        self.assertEqual(data['mr'].shape, (100, 100, 1, 1, 1))

    def test_renaming(self):
        data_param = {'mr': {'path_to_search': IMAGE_PATH_2D}}
        group_param = {'ct': ('mr',)}
        reader = ImageReader().initialise(data_param, group_param)
        self.renamed_property_asserts(reader)
        idx, data, interp = reader()

        # test output
        self.assertTrue('ct' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'ct': (1,)})
        self.assertEqual(data['ct'].shape, (100, 100, 1, 1, 1))

    def test_reader_field(self):
        data_param = {'mr': {'path_to_search': IMAGE_PATH_2D}}
        group_param = {'ct': ('mr',)}
        reader = ImageReader(['ct']).initialise(data_param, group_param)
        self.renamed_property_asserts(reader)
        idx, data, interp = reader()

        # test output
        self.assertTrue('ct' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'ct': (1,)})
        self.assertEqual(data['ct'].shape, (100, 100, 1, 1, 1))

        with self.assertRaisesRegexp(ValueError, ''):
            # grouping name 'ct' but
            reader = ImageReader(['mr']).initialise(data_param, group_param)

    def test_input_properties(self):
        data_param = {'mr': {'path_to_search': IMAGE_PATH_2D,
                             'csv_file': '2d_test.csv'}}
        reader = ImageReader().initialise(data_param)
        self.default_property_asserts(reader)
        idx, data, interp = reader()

        # test output
        self.assertTrue('mr' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'mr': (1,)})
        self.assertEqual(data['mr'].shape, (100, 100, 1, 1, 1))

    def test_no_2d_resampling_properties(self):
        data_param = {'mr': {'path_to_search': IMAGE_PATH_2D,
                             'csv_file': '2d_test.csv',
                             'pixdim': (2, 2, 2),
                             'axcodes': 'RAS'}}
        reader = ImageReader().initialise(data_param)
        self.assertEqual(reader.output_list[0]['mr'].output_pixdim, (None,))
        self.assertEqual(reader.output_list[0]['mr'].output_axcodes, (None,))
        idx, data, interp = reader()

        # test output
        self.assertTrue('mr' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'mr': (1,)})
        self.assertEqual(data['mr'].shape, (100, 100, 1, 1, 1))

    def test_2d_as_5D_multimodal_properties(self):
        data_param = {'mr': {'path_to_search': IMAGE_PATH_2D,
                             'filename_contains': '_u',
                             'pixdim': (2, 2, 2),
                             'axcodes': 'RAS'}}
        grouping_param = {'ct': ('mr', 'mr', 'mr')}
        reader = ImageReader().initialise(data_param, grouping_param)
        self.assertEqual(reader.spatial_ranks, {'ct': 2})

    def test_2D_multimodal_properties(self):
        data_param = {'mr': {'path_to_search': IMAGE_PATH_2D,
                             'filename_contains': '_g',
                             'pixdim': (2, 1.5, 2),
                             'axcodes': 'RAS'}}
        grouping_param = {'ct': ('mr', 'mr', 'mr')}
        reader = ImageReader().initialise(data_param, grouping_param)
        self.assertDictEqual(reader.spatial_ranks, {'ct': 2})
        self.assertEqual(reader.output_list[0]['ct'].output_pixdim,
                         ((2.0, 1.5, 2.0),) * 3)
        self.assertEqual(reader.output_list[0]['ct'].output_axcodes,
                         (('R', 'A', 'S'),) * 3)

        # test output
        idx, data, interp = reader()
        self.assertTrue('ct' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'ct': (1, 1, 1)})
        self.assertEqual(data['ct'].shape, (100, 100, 1, 1, 3))


class Read2D_1DTest(NiftyNetTestCase):
    # loading 2d images of rank 3: [x, y, 1]
    def test_no_2d_resampling_properties(self):
        data_param = {'mr': {'path_to_search': IMAGE_PATH_2D_1,
                             'csv_file': '2d_test.csv',
                             'filename_contains': '_img',
                             'pixdim': (2, 2, 2),
                             'axcodes': 'RAS'}}
        reader = ImageReader().initialise(data_param)
        self.assertEqual(reader.output_list[0]['mr'].output_pixdim,
                         ((2.0, 2.0, 2.0),))
        self.assertEqual(reader.output_list[0]['mr'].output_axcodes,
                         (('R', 'A', 'S'),))
        idx, data, interp = reader()

        # test output
        self.assertTrue('mr' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'mr': (1,)})
        self.assertEqual(data['mr'].shape, (120, 160, 1, 1, 1))

    def test_2D_multimodal_properties(self):
        data_param = {'mr': {'path_to_search': IMAGE_PATH_2D_1,
                             'filename_contains': '_img',
                             'pixdim': (2, 1.5, 2),
                             'axcodes': 'RAS'}}
        grouping_param = {'ct': ('mr', 'mr', 'mr')}
        reader = ImageReader().initialise(data_param, grouping_param)
        self.assertDictEqual(reader.spatial_ranks, {'ct': 2})
        self.assertEqual(reader.output_list[0]['ct'].output_pixdim,
                         ((2.0, 1.5, 2.0),) * 3)
        self.assertEqual(reader.output_list[0]['ct'].output_axcodes,
                         (('R', 'A', 'S'),) * 3)

        # test output
        idx, data, interp = reader()
        self.assertTrue('ct' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'ct': (1, 1, 1)})
        self.assertEqual(data['ct'].shape, (120, 160, 1, 1, 3))


class Read2D_1D_x1y_Test(NiftyNetTestCase):
    # loading 2d images of rank 3: [x, 1, y]
    def test_no_2d_resampling_properties(self):
        data_param = {'mr': {'path_to_search': IMAGE_PATH_2D_2,
                             'filename_contains': 'x_1_y',
                             'pixdim': (2, 2, 2),
                             'axcodes': 'RAS'}}
        reader = ImageReader().initialise(data_param)
        self.assertEqual(reader.output_list[0]['mr'].output_pixdim,
                         ((2.0, 2.0, 2.0),))
        self.assertEqual(reader.output_list[0]['mr'].output_axcodes,
                         (('R', 'A', 'S'),))
        idx, data, interp = reader()

        # test output
        self.assertTrue('mr' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'mr': (1,)})
        self.assertEqual(data['mr'].shape, (100, 100, 1, 1, 1))

    def test_2D_multimodal_properties(self):
        data_param = {'mr': {'path_to_search': IMAGE_PATH_2D_2,
                             'filename_contains': 'x_1_y',
                             'pixdim': (2, 1.5, 2),
                             'axcodes': 'RAS'}}
        grouping_param = {'ct': ('mr', 'mr', 'mr')}
        reader = ImageReader().initialise(data_param, grouping_param)
        self.assertDictEqual(reader.spatial_ranks, {'ct': 2})
        self.assertEqual(reader.output_list[0]['ct'].output_pixdim,
                         ((2.0, 1.5, 2.0),) * 3)
        self.assertEqual(reader.output_list[0]['ct'].output_axcodes,
                         (('R', 'A', 'S'),) * 3)

        # test output
        idx, data, interp = reader()
        self.assertTrue('ct' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'ct': (1, 1, 1)})
        self.assertEqual(data['ct'].shape, (100, 100, 1, 1, 3))


class Read2D_colorTest(NiftyNetTestCase):
    # loading 2d images of rank 3: [x, y, 3] or [x, y, 4]
    def test_no_2d_resampling_properties(self):
        data_param = {'mr': {'path_to_search': IMAGE_PATH_2D,
                             'csv_file': '2d_test.csv',
                             'filename_contains': '_u',
                             'pixdim': (2, 2, 2),
                             'axcodes': 'RAS'}}
        reader = ImageReader().initialise(data_param)
        self.assertEqual(reader.output_list[0]['mr'].output_pixdim,
                         ((2.0, 2.0, 2.0),))
        self.assertEqual(reader.output_list[0]['mr'].output_axcodes,
                         (('R', 'A', 'S'),))
        idx, data, interp = reader()

        # test output
        self.assertTrue('mr' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'mr': (1,)})
        self.assertEqual(data['mr'].shape, (100, 100, 1, 1, 3))

    def test_2D_multimodal_properties(self):
        data_param = {
            'mr': {'path_to_search': IMAGE_PATH_2D,
                   'filename_contains': '_u',
                   'pixdim': (2, 1.5, 2),
                   'axcodes': 'RAS'}}
        grouping_param = {'ct': ('mr', 'mr', 'mr')}
        reader = ImageReader().initialise(data_param, grouping_param)
        self.assertDictEqual(reader.spatial_ranks, {'ct': 2})
        self.assertEqual(reader.output_list[0]['ct'].output_pixdim,
                         ((2.0, 1.5, 2.0),) * 3)
        self.assertEqual(reader.output_list[0]['ct'].output_axcodes,
                         (('R', 'A', 'S'),) * 3)

        # test output
        idx, data, interp = reader()
        self.assertTrue('ct' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'ct': (1, 1, 1)})
        self.assertEqual(data['ct'].shape, (100, 100, 1, 1, 9))


class Read3DTest(NiftyNetTestCase):
    # loading 3d images of rank 3: [x, y, z]
    def test_3d_resampling_properties(self):
        data_param = {
            'mr': {'path_to_search': IMAGE_PATH_3D,
                   'filename_contains': 'Lesion',
                   'pixdim': (4, 3, 4),
                   'axcodes': 'RAS'}}
        reader = ImageReader().initialise(data_param)
        self.assertDictEqual(reader.spatial_ranks, {'mr': 3})
        self.assertEqual(reader.output_list[0]['mr'].output_pixdim,
                         ((4.0, 3.0, 4.0),))
        self.assertEqual(reader.output_list[0]['mr'].output_axcodes,
                         (('R', 'A', 'S'),))
        idx, data, interp = reader()

        # test output
        self.assertTrue('mr' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'mr': (1,)})
        # allows rounding error spatially
        self.assertAllClose(data['mr'].shape[:3], (62, 83, 62), atol=1)
        self.assertAllClose(data['mr'].shape[3:], (1, 1))

    def test_3d_multiple_properties(self):
        """
        loading two modalities, grouping subject names only
        """
        data_param = {
            'mr': {'path_to_search': IMAGE_PATH_3D,
                   'filename_contains': 'Lesion',
                   'pixdim': (4, 3, 4),
                   'axcodes': 'RAS'},
            'ct': {'path_to_search': IMAGE_PATH_3D,
                   'filename_contains': 'Lesion',
                   'pixdim': (4, 3, 4),
                   'axcodes': 'RAS'}}
        reader = ImageReader().initialise(data_param)
        self.assertDictEqual(reader.spatial_ranks, {'mr': 3, 'ct': 3})
        self.assertEqual(reader.output_list[0]['mr'].output_pixdim,
                         ((4.0, 3.0, 4.0),))
        self.assertEqual(reader.output_list[0]['mr'].output_axcodes,
                         (('R', 'A', 'S'),))
        idx, data, interp = reader()

        # test output
        self.assertTrue('mr' in data)
        self.assertTrue('ct' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'mr': (1,), 'ct': (1,)})
        # allows rounding error spatially
        self.assertAllClose(data['mr'].shape[:3], (62, 83, 62), atol=1)
        self.assertAllClose(data['mr'].shape[3:], (1, 1))
        self.assertAllClose(data['ct'].shape[:3], (62, 83, 62), atol=1)
        self.assertAllClose(data['ct'].shape[3:], (1, 1))

    def test_3d_concat_properties(self):
        """
        loading two modalities, grouping subject names only
        """
        data_param = {
            'mr': {'path_to_search': IMAGE_PATH_3D,
                   'filename_contains': 'LesionFin',
                   'pixdim': (4, 3, 4),
                   'axcodes': 'RAS'},
            'ct': {'path_to_search': IMAGE_PATH_3D,
                   'filename_contains': 'FLAIR',
                   'pixdim': (4, 3, 4),
                   'axcodes': 'RAS'}}
        grouping_param = {'image': ('mr', 'ct')}
        reader = ImageReader().initialise(data_param, grouping_param)
        self.assertDictEqual(reader.spatial_ranks, {'image': 3})
        self.assertEqual(reader.output_list[0]['image'].output_pixdim,
                         ((4.0, 3.0, 4.0),) * 2)
        self.assertEqual(reader.output_list[0]['image'].output_axcodes,
                         (('R', 'A', 'S'),) * 2)
        idx, data, interp = reader()

        # test output
        self.assertTrue('image' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'image': (1, 1)})
        # allows rounding error spatially
        self.assertAllClose(data['image'].shape[:3], (62, 83, 62), atol=1)
        self.assertAllClose(data['image'].shape[3:], (1, 2))


class Read3D_1_1_Test(NiftyNetTestCase):
    # loading 5d images of rank 3: [x, y, z, 1, 1]
    def test_3d_resampling_properties(self):
        data_param = {
            'mr': {'path_to_search': IMAGE_PATH_3D_1,
                   'filename_contains': 'x_y_z_1_1',
                   'pixdim': (4, 3, 4),
                   'axcodes': 'RAS'}}
        reader = ImageReader().initialise(data_param)
        self.assertDictEqual(reader.spatial_ranks, {'mr': 3})
        self.assertEqual(reader.output_list[0]['mr'].output_pixdim,
                         ((4.0, 3.0, 4.0),))
        self.assertEqual(reader.output_list[0]['mr'].output_axcodes,
                         (('R', 'A', 'S'),))
        idx, data, interp = reader()

        # test output
        self.assertTrue('mr' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'mr': (1,)})
        # allows rounding error spatially
        self.assertAllClose(data['mr'].shape[:3], (12, 8, 10), atol=1)
        self.assertAllClose(data['mr'].shape[3:], (1, 1))

    def test_3d_multiple_properties(self):
        """
        loading two modalities, grouping subject names only
        """
        data_param = {
            'mr': {'path_to_search': IMAGE_PATH_3D_1,
                   'filename_contains': 'x_y_z_1_1',
                   'pixdim': (4, 3, 4),
                   'axcodes': 'RAS'},
            'ct': {'path_to_search': IMAGE_PATH_3D_1,
                   'filename_contains': 'x_y_z_1_1',
                   'pixdim': (4, 3, 4),
                   'axcodes': 'RAS'}}
        reader = ImageReader().initialise(data_param)
        self.assertDictEqual(reader.spatial_ranks, {'mr': 3, 'ct': 3})
        self.assertEqual(reader.output_list[0]['mr'].output_pixdim,
                         ((4.0, 3.0, 4.0),))
        self.assertEqual(reader.output_list[0]['mr'].output_axcodes,
                         (('R', 'A', 'S'),))
        idx, data, interp = reader()

        # test output
        self.assertTrue('mr' in data)
        self.assertTrue('ct' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'mr': (1,), 'ct': (1,)})
        # allows rounding error spatially
        self.assertAllClose(data['mr'].shape[:3], (12, 8, 10), atol=1)
        self.assertAllClose(data['mr'].shape[3:], (1, 1))
        self.assertAllClose(data['ct'].shape[:3], (12, 8, 10), atol=1)
        self.assertAllClose(data['ct'].shape[3:], (1, 1))

    def test_3d_concat_properties(self):
        """
        loading two modalities, grouping subject names only
        """
        data_param = {
            'mr': {'path_to_search': IMAGE_PATH_3D_1,
                   'filename_contains': 'x_y_z_1_1',
                   'pixdim': (4, 3, 4),
                   'axcodes': 'RAS'},
            'ct': {'path_to_search': IMAGE_PATH_3D_1,
                   'filename_contains': 'x_y_z_1_1',
                   'pixdim': (4, 3, 4),
                   'axcodes': 'RAS'}}
        grouping_param = {'image': ('mr', 'ct')}
        reader = ImageReader().initialise(data_param, grouping_param)
        self.assertDictEqual(reader.spatial_ranks, {'image': 3})
        self.assertEqual(reader.output_list[0]['image'].output_pixdim,
                         ((4.0, 3.0, 4.0),) * 2)
        self.assertEqual(reader.output_list[0]['image'].output_axcodes,
                         (('R', 'A', 'S'),) * 2)
        idx, data, interp = reader()

        # test output
        self.assertTrue('image' in data)
        self.assertTrue(idx in range(len(reader.output_list)))
        self.assertDictEqual(interp, {'image': (1, 1)})
        # allows rounding error spatially
        self.assertAllClose(data['image'].shape[:3], (12, 8, 10), atol=1)
        self.assertAllClose(data['image'].shape[3:], (1, 2))


if __name__ == "__main__":
    tf.test.main()
