# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from niftynet.utilities.download import download
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig
from tests.niftynet_testcase import NiftyNetTestCase

MODEL_HOME = NiftyNetGlobalConfig().get_niftynet_home_folder()

TEST_CASE_1 = 'dense_vnet_abdominal_ct_model_zoo'
TEST_CASE_1_TARGET = os.path.join(
    MODEL_HOME, 'models', 'dense_vnet_abdominal_ct')
TEST_CASE_2 = 'default'
TEST_CASE_2_TARGET = os.path.join(MODEL_HOME, 'examples', TEST_CASE_2)
TEST_CASE_3 = 'default_multimodal'
TEST_CASE_3_TARGET = os.path.join(MODEL_HOME, 'examples', TEST_CASE_3)

TEST_WRONG_ID = '42'


class NetDownloadTest(NiftyNetTestCase):
    def test_download(self):
        self.assertTrue(download(TEST_CASE_1, False))
        self.assertTrue(os.path.isdir(TEST_CASE_1_TARGET))

        if os.path.isdir(TEST_CASE_1_TARGET):
            print('skipping tests: %s folder exists' % TEST_CASE_1_TARGET)
        else:
            self.assertTrue(download(TEST_CASE_1, True))
            self.assertTrue(os.path.isdir(TEST_CASE_1_TARGET))

    def test_wrong_ids(self):
        self.assertFalse(download([], False))
        self.assertFalse(download((), False))
        self.assertFalse(download(None, False))
        self.assertFalse(download([], True))
        self.assertFalse(download((), True))
        self.assertFalse(download(None, True))
        self.assertFalse(download(TEST_WRONG_ID, True))
        self.assertFalse(download(TEST_WRONG_ID, False))

    def test_multiple_ids(self):
        self.assertTrue(
            download([TEST_CASE_1, TEST_CASE_2, TEST_CASE_3], False))
        self.assertTrue(os.path.isdir(TEST_CASE_1_TARGET))
        self.assertTrue(os.path.isdir(TEST_CASE_2_TARGET))
        self.assertTrue(os.path.isdir(TEST_CASE_3_TARGET))


if __name__ == "__main__":
    tf.test.main()
