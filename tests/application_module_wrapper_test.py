# -*- coding: utf-8 -*-
"""
Unit test module for ApplicationModuleWrapper
"""
from __future__ import absolute_import

import unittest, os
from configparser import ConfigParser
from glob import glob

import nibabel as nib
import tensorflow as tf

from niftynet.application.application_module_wrapper import \
    RegressionApplicationModule
from niftynet.io.image_sets_partitioner import (
    COLUMN_PHASE, COLUMN_UNIQ_ID)
from niftynet.utilities.download import download
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig

MODEL_HOME = NiftyNetGlobalConfig().get_niftynet_home_folder()


def get_input_path(config_file, image_type):
    parser = ConfigParser()
    parser.read(config_file)

    return os.path.join(MODEL_HOME, parser.get(image_type, 'path_to_search'))


def get_file_list(config_file, image_type):
    input_dir = get_input_path(config_file, image_type)

    return [img_path for img_path in glob(os.path.join(input_dir, '*.nii.gz'))]


@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class ApplicationModuleWrapperTest(tf.test.TestCase):
    """
    Adaptation of tests.test_model_zoo.MRCTRegressionModelZooTest
    for testing of ApplicationModuleWrapper
    """

    id = 'autocontext_mr_ct_model_zoo'
    location = 'autocontext_mr_ct'
    application = 'net_regress'
    config = os.path.join(MODEL_HOME, 'extensions', 'autocontext_mr_ct', 'net_autocontext.ini')
    expected_output_train = [
        os.path.join('error_maps','WEB.nii.gz'),
        ]
    expected_output_inference = [
        os.path.join('autocontext_output','CHA_niftynet_out.nii.gz'),
        ]
    modality1_images = []
    modality2_images = []
    output_received = False

    def setUp(self):
        tf.test.TestCase.setUp(self)
        download(self.id, download_if_already_existing=True, verbose=False)

    def get_modality1_image(self, idx):
        return nib.load(self.modality1_images[idx]).get_data()

    def get_modality2_image(self, idx):
        return nib.load(self.modality2_images[idx]).get_data()

    def output_callback(self, outpt, id, inpt):
        self.output_received = True
        self.assertAllClose(
            inpt.flatten(), self.get_modality1_image(int(id)).flatten())

    FAKE_CSV_PATH = os.path.join('testing_data', 'app_module_fake_split.csv')

    def fake_split(self):
        nof_train = 2*len(self.modality1_images)/3
        with open(self.FAKE_CSV_PATH, 'w') as fout:
            fout.write('%s,%s\n' % (COLUMN_UNIQ_ID, COLUMN_PHASE))
            for idx in range(len(self.modality1_images)):
                fout.write('%i,%s\n' % (idx, 'training' if idx < nof_train else 'inference'))

    def test_infer_mem_io(self):
        modality1_images = get_file_list(self.config, 'MODALITY1')
        modality2_path = get_input_path(self.config, 'MODALITY2')

        self.modality2_images, self.modality1_images = ([], [])
        for mod1 in modality1_images:
            mod2 = os.path.join(modality2_path, os.path.basename(mod1))
            if os.path.isfile(mod2):
                self.modality1_images.append(mod1)
                self.modality2_images.append(mod2)
        assert len(self.modality1_images) > 0

        app_module = RegressionApplicationModule(self.config)
        self.fake_split()

        app_module.set_input_callback('MODALITY1', self.get_modality1_image)\
                  .set_input_callback('MODALITY2', self.get_modality2_image)\
                  .set_num_subjects(len(self.modality1_images))\
                  .override_params('TRAINING', starting_iter=0, max_iter=2, validation_every_n=-1)\
                  .override_params('SYSTEM', dataset_split_file=self.FAKE_CSV_PATH)\
                  .set_action('train')\
                  .initialise_application()\
                  .run()

        checkpoint = os.path.join(MODEL_HOME, 'models', self.location, 'models', 'model.ckpt-2.index')
        self.assertTrue(os.path.exists(checkpoint))

        self.output_received = False
        app_module.set_output_callback(self.output_callback)\
                  .set_action('inference')\
                  .initialise_application()\
                  .run()

        for eo in self.expected_output_train:
            output = os.path.join(MODEL_HOME, 'models', self.location, eo)
            self.assertTrue(os.path.exists(output), 'Expected {} to exist.'.format(output))


if __name__ == "__main__":
    tf.test.main()
