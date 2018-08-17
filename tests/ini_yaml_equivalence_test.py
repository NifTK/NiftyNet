from unittest import TestCase
import os


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
        # TODO
        raise NotImplementedError

    def test_read_ini_equivalent_to_read_yaml(self):
        # TODO
        raise NotImplementedError

    def test_incompatible_file_not_read(self):
        # TODO
        raise NotImplementedError

    def test_sections_same(self):
        # TODO
        raise NotImplementedError

    def test_items_same(self):
        # TODO
        raise NotImplementedError

    def test_add_section_same(self):
        # TODO
        raise NotImplementedError

    def test_set_same(self):
        # TODO
        raise NotImplementedError

    def test_remove_section_same(self):
        # TODO
        raise NotImplementedError

    def has_section_same(self):
        # TODO
        raise NotImplementedError
