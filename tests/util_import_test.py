# -*- coding: utf-8 -*-
import tensorflow as tf

from niftynet.utilities.util_import import require_module
from tests.niftynet_testcase import NiftyNetTestCase

class OptionalPackageTest(NiftyNetTestCase):
    def test_installed(self):
        require_module('tensorflow')

    def test_installed_min_version(self):
        require_module('tensorflow', 1.0)

    def test_no_package(self):
        with self.assertRaisesRegexp(ImportError, ''):
            require_module('foobar_wrong_case', mandatory=True)

    def test_wrong_version(self):
        with self.assertRaisesRegexp(AssertionError, ''):
            require_module('tensorflow', 100, mandatory=True)

    def test_self_version(self):
        require_module('importlib')

    def test_no_version_info(self):
        require_module('importlib', 0)

    def test_no_input(self):
        with self.assertRaisesRegexp(ImportError, ''):
            require_module([], mandatory=True)
        with self.assertRaisesRegexp(ImportError, ''):
            require_module(None, mandatory=True)


if __name__ == "__main__":
    tf.test.main()
