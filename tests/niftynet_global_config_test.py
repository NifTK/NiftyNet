from unittest import TestCase
from os.path import (expanduser, join, isdir, isfile)
from os import remove
from shutil import rmtree
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig


class NiftyNetGlobalConfigTest(TestCase):
    """For reliably testing the global config file, the tests are grouped
    and ordered by including a number.

    https://docs.python.org/2/library/unittest.html says: "Note that the
    order in which the various test cases will be run is determined by
    sorting the test function names with respect to the built-in ordering
    for strings."
    """

    @classmethod
    def typify(cls, file_path):
        """Append file type extension to passed file path."""
        return '.'.join([file_path, cls.file_type])

    @classmethod
    def remove_path(cls, path):
        """Remove passed item, whether it's a file or directory."""
        if isdir(path):
            rmtree(path)
        elif isfile(path):
            remove(path)

    @classmethod
    def setUpClass(cls):
        cls.config_home = join(expanduser('~'), '.niftynet')
        cls.file_type = 'yml'
        cls.config_file = join(cls.config_home, cls.typify('config'))

        cls.default_config_opts = {
            'niftynet_home': join(expanduser('~'), 'niftynet')
        }

    @classmethod
    def tearDownClass(cls):
        # TODO
        pass

    def setUp(self):
        NiftyNetGlobalConfigTest.remove_path(NiftyNetGlobalConfigTest.config_home)
        NiftyNetGlobalConfigTest.remove_path(
            NiftyNetGlobalConfigTest.default_config_opts['niftynet_home']
        )

    def tearDown(self):
        self.setUp()

    def test_000_global_config_singleton(self):
        global_config_1 = NiftyNetGlobalConfig()
        global_config_2 = NiftyNetGlobalConfig()
        self.assertEqual(global_config_1, global_config_2)
        self.assertTrue(global_config_1 is global_config_2)

    def test_010_non_existing_config_file_created(self):
        global_config = NiftyNetGlobalConfig()
        self.assertTrue(isfile(NiftyNetGlobalConfigTest.config_file))
        self.assertEqual(global_config.get_niftynet_config_folder(),
                         NiftyNetGlobalConfigTest.config_home)

    def test_011_existing_config_file_loaded(self):
        # TODO
        self.fail('not implemented')

    def test_012_incorrect_config_file_backed_up(self):
        # TODO
        self.fail('not implemented')
