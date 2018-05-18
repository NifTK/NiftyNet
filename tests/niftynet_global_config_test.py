from glob import glob
from os import (remove, makedirs, environ)
from os.path import (expanduser, join, isdir, isfile)
from os.path import getmtime
from shutil import rmtree
from unittest import (TestCase, skipUnless)

from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig


class NiftyNetGlobalConfigTest(TestCase):
    """Tests included here all pertain to the NiftyNet global configuration
    file and `NiftyNetGlobalConfig` is a singleton. These require each test
    to be run separately. This is why all tests are decorated with
    `skipUnless`.
    """

    @classmethod
    def typify(cls, file_path):
        """Append file type extension to passed file path."""
        return '.'.join([file_path, cls.file_type])

    @classmethod
    def remove_path(cls, path):
        """Remove passed item, whether it's a file or directory."""
        print("removing {}".format(path))
        if isdir(path):
            rmtree(path)
        elif isfile(path):
            remove(path)

    @classmethod
    def setUpClass(cls):
        cls.config_home = join(expanduser('~'), '.niftynet')
        cls.file_type = 'ini'
        cls.config_file = join(cls.config_home, cls.typify('config'))

        cls.header = '[global]'
        cls.default_config_opts = {
            'home': '~/niftynet',
            'ext': 'extensions',
            'ext_mods': ['network']
        }

    def setUp(self):
        NiftyNetGlobalConfigTest.remove_path(
            NiftyNetGlobalConfigTest.config_home)
        NiftyNetGlobalConfigTest.remove_path(
            expanduser(NiftyNetGlobalConfigTest.default_config_opts['home'])
        )

    def tearDown(self):
        self.setUp()

    @skipUnless('GLOBAL_CONFIG_TEST_gcs' in environ,
                'set GLOBAL_CONFIG_TEST_gcs to run')
    def test_global_config_singleton(self):
        global_config_1 = NiftyNetGlobalConfig()
        global_config_2 = NiftyNetGlobalConfig()
        self.assertEqual(global_config_1, global_config_2)
        self.assertTrue(global_config_1 is global_config_2)

    @skipUnless('GLOBAL_CONFIG_TEST_necfc' in environ,
                'set GLOBAL_CONFIG_TEST_necfc to run')
    def test_non_existing_config_file_created(self):
        self.assertFalse(isfile(NiftyNetGlobalConfigTest.config_file))
        global_config = NiftyNetGlobalConfig().setup()
        self.assertTrue(isfile(NiftyNetGlobalConfigTest.config_file))
        self.assertEqual(global_config.get_niftynet_config_folder(),
                         NiftyNetGlobalConfigTest.config_home)

    @skipUnless('GLOBAL_CONFIG_TEST_ecfl' in environ,
                'set GLOBAL_CONFIG_TEST_ecfl to run')
    def test_existing_config_file_loaded(self):
        # create a config file with a custom NiftyNet home
        makedirs(NiftyNetGlobalConfigTest.config_home)
        custom_niftynet_home = '~/customniftynethome'
        custom_niftynet_home_abs = expanduser(custom_niftynet_home)
        config = ''.join(['home = ', custom_niftynet_home])
        with open(NiftyNetGlobalConfigTest.config_file, 'w') as config_file:
            config_file.write('\n'.join(
                [NiftyNetGlobalConfigTest.header, config]))

        global_config = NiftyNetGlobalConfig().setup()
        self.assertEqual(global_config.get_niftynet_home_folder(),
                         custom_niftynet_home_abs)
        NiftyNetGlobalConfigTest.remove_path(custom_niftynet_home_abs)

    @skipUnless('GLOBAL_CONFIG_TEST_icfbu' in environ,
                'set GLOBAL_CONFIG_TEST_icfbu to run')
    def test_incorrect_config_file_backed_up(self):
        # create an incorrect config file at the correct location
        makedirs(NiftyNetGlobalConfigTest.config_home)
        incorrect_config = '\n'.join([NiftyNetGlobalConfigTest.header,
                                      'invalid_home_tag = ~/niftynet'])
        with open(NiftyNetGlobalConfigTest.config_file, 'w') as config_file:
            config_file.write(incorrect_config)

        # the following should back it up and replace it with default config
        global_config = NiftyNetGlobalConfig().setup()

        self.assertTrue(isfile(NiftyNetGlobalConfigTest.config_file))
        self.assertEqual(global_config.get_niftynet_config_folder(),
                         NiftyNetGlobalConfigTest.config_home)

        # check if incorrect file was backed up
        found_files = glob(
            join(NiftyNetGlobalConfigTest.config_home,
                 NiftyNetGlobalConfigTest.typify('config-backup-*')))
        self.assertTrue(len(found_files) == 1)
        with open(found_files[0], 'r') as backup_file:
            self.assertEqual(backup_file.read(), incorrect_config)

        # cleanup: remove backup file
        NiftyNetGlobalConfigTest.remove_path(found_files[0])

    @skipUnless('GLOBAL_CONFIG_TEST_nenhc' in environ,
                'set GLOBAL_CONFIG_TEST_nenhc to run')
    def test_non_existing_niftynet_home_created(self):
        niftynet_home = expanduser(
            NiftyNetGlobalConfigTest.default_config_opts['home'])
        NiftyNetGlobalConfigTest.remove_path(niftynet_home)
        self.assertFalse(isdir(niftynet_home))
        niftynet_ext = join(
            niftynet_home, NiftyNetGlobalConfigTest.default_config_opts['ext']
        )
        self.assertFalse(isfile(join(niftynet_ext, '__init__.py')))
        for mod in NiftyNetGlobalConfigTest.default_config_opts['ext_mods']:
            self.assertFalse(isfile(join(niftynet_ext, mod, '__init__.py')))

        global_config = NiftyNetGlobalConfig().setup()

        self.assertTrue(isdir(niftynet_home))
        self.assertTrue(isfile(join(niftynet_ext, '__init__.py')))
        for mod in NiftyNetGlobalConfigTest.default_config_opts['ext_mods']:
            self.assertTrue(isfile(join(niftynet_ext, mod, '__init__.py')))

    @skipUnless('GLOBAL_CONFIG_TEST_enhnt' in environ,
                'set GLOBAL_CONFIG_TEST_enhnt to run')
    def test_existing_niftynet_home_not_touched(self):
        niftynet_home = expanduser(
            NiftyNetGlobalConfigTest.default_config_opts['home'])
        makedirs(niftynet_home)
        niftynet_ext = join(
            niftynet_home, NiftyNetGlobalConfigTest.default_config_opts['ext']
        )
        makedirs(niftynet_ext)
        niftynet_ext_init = join(niftynet_ext, '__init__.py')
        open(niftynet_ext_init, 'w').close()
        mtime_before = getmtime(niftynet_ext_init)

        global_config = NiftyNetGlobalConfig()

        mtime_after = getmtime(niftynet_ext_init)
        self.assertEqual(mtime_before, mtime_after)
