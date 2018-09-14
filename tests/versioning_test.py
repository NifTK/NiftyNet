# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.utilities.versioning import check_pep_440
from niftynet.utilities.versioning import get_niftynet_version_string


class VersioningTest(tf.test.TestCase):
    def test_version(self):
        version_str = get_niftynet_version_string()
        expected_string = "NiftyNet version "
        self.assertEqual(version_str[:len(expected_string)], expected_string)

        check_pep_440()


if __name__ == "__main__":
    tf.test.main()
