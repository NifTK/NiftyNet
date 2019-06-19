# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys

import tensorflow as tf

import net_autoencoder
import net_gan
import net_regress
import net_run
import net_segment

from tests.niftynet_testcase import NiftyNetTestCase

class EntryPointTest(NiftyNetTestCase):
    def test_wrong_app(self):
        sys.argv = ['', 'train',
                    '-a', 'foo',
                    '-c', os.path.join('config', 'default_segmentation.ini')]
        with self.assertRaisesRegexp(ValueError, 'application'):
            net_run.main()

        sys.argv = ['', 'train',
                    '-c', os.path.join('config', 'default_segmentation.ini')]
        with self.assertRaisesRegexp(ValueError, 'application'):
            net_run.main()

    def test_wrong_config(self):
        sys.argv = ['', 'train',
                    '-a', 'net_segment',
                    '-c', os.path.join('foo', 'default_segmentation.ini')]
        with self.assertRaisesRegexp(IOError, ''):
            net_run.main()

        sys.argv = ['', 'train',
                    '-a', 'net_segment']
        with self.assertRaisesRegexp(IOError, ''):
            net_run.main()

    def test_no_action(self):
        sys.argv = ['',
                    '-a', 'net_segment',
                    '-c', os.path.join('config', 'default_segmentation.ini')]
        with self.assertRaisesRegexp(SystemExit, ''):
            net_run.main()

    def test_wrong_param(self):
        sys.argv = ['',
                    '-a', 'net_segment',
                    '-c', os.path.join('config', 'default_segmentation.ini'),
                    '--foo=bar']
        with self.assertRaisesRegexp(SystemExit, ''):
            net_run.main()

    def test_empty(self):
        with self.assertRaisesRegexp(SystemExit, ''):
            net_run.main()
        with self.assertRaisesRegexp(SystemExit, ''):
            net_gan.main()
        with self.assertRaisesRegexp(SystemExit, ''):
            net_segment.main()
        with self.assertRaisesRegexp(SystemExit, ''):
            net_regress.main()
        with self.assertRaisesRegexp(SystemExit, ''):
            net_autoencoder.main()


if __name__ == "__main__":
    tf.test.main()
