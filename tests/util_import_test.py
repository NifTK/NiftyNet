# -*- coding: utf-8 -*-
import tensorflow as tf

from niftynet.utilities.util_import import check_module


class OptionalPackageTest(tf.test.TestCase):
    def test_installed(self):
        check_module('tensorflow')

    def test_installed_min_version(self):
        check_module('tensorflow', 1.0)

    def test_no_package(self):
        with self.assertRaisesRegexp(ImportError, ''):
            check_module('foobar_wrong_case')

    def test_wrong_version(self):
        with self.assertRaisesRegexp(AssertionError, ''):
            check_module('tensorflow', 100)

    def test_self_version(self):
        check_module('importlib')

    def test_no_version_info(self):
        check_module('importlib', 0)

    def test_no_input(self):
        with self.assertRaisesRegexp(ImportError, ''):
            check_module([])
        with self.assertRaisesRegexp(ImportError, ''):
            check_module(None)


if __name__ == "__main__":
    tf.test.main()
