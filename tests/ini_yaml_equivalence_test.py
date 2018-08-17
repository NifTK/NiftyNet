from unittest import TestCase
import os
from niftynet.utilities import NiftyNetLaunchConfig


class IniYamlEquivalenceTest(TestCase):

    @classmethod
    def setUpClass(cls):
        config_files_dir = os.path.join(
            os.path.split(__file__)[0], '..', 'config'
        )
        config_file = 'default_segmentation'
        cls.ini_file = os.path.join(
            config_files_dir, '{}.ini'.format(config_file)
        )
        cls.yaml_file = os.path.join(
            config_files_dir, '{}.yml'.format(config_file)
        )

    def setUp(self):
        self.ini_config = NiftyNetLaunchConfig()
        self.ini_config.read(IniYamlEquivalenceTest.ini_file)

        self.yaml_config = NiftyNetLaunchConfig()
        self.yaml_config.read(IniYamlEquivalenceTest.yaml_config)

    def test_items_same(self):
        ini_items = self.ini_config.items()
        self.assertIsNotNone(ini_items)
        yaml_items = self.yaml_config.items()
        self.assertIsNotNone(yaml_items)
        self.assertFalse(yaml_items == dict())
        self.assertTrue(yaml_items == ini_items)

    def test_incompatible_file_not_read(self):
        with self.assertRaises(ValueError):
            self.ini_config.read(IniYamlEquivalenceTest.yaml_file)

        with self.assertRaises(ValueError):
            self.yaml_config.read(IniYamlEquivalenceTest.ini_file)

    def test_sections_same(self):
        ini_sections = self.ini_config.sections()
        yaml_sections = self.yaml_config.sections()
        self.assertIsNotNone(yaml_sections)
        self.assertFalse(yaml_sections == dict())
        self.assertTrue(yaml_sections == ini_sections)

    def test_add_section_same(self):
        new_section = 'new_section_that_should_not_exist'
        self.ini_config.add_section(new_section)
        self.assertIn(new_section, self.ini_config.sections())
        self.yaml_config.add_section(new_section)
        self.assertTrue(self.yaml_config.items() == self.ini_config.items())

    def test_set_same(self):
        # TODO
        raise NotImplementedError

    def test_remove_section_same(self):
        section_to_remove = self.ini_config.sections()[0]
        self.ini_config.remove_section(section_to_remove)
        self.yaml_config.remove_section(section_to_remove)
        self.assertTrue(self.yaml_config.items() == self.ini_config.items())

    def test_has_section_same(self):
        # TODO
        raise NotImplementedError
