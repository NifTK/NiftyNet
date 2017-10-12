from unittest import TestCase
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig
from niftynet.utilities.decorators import singleton


class NiftyNetGlobalConfigTest(TestCase):

    def test_global_config_singleton(self):
        global_config_1 = NiftyNetGlobalConfig()
        global_config_2 = NiftyNetGlobalConfig()
        self.assertEqual(global_config_1, global_config_2)


@singleton
class Dummy(object):
    def __init__(self):
        pass


if __name__ == '__main__':
    dummy_1 = Dummy()
    dummy_2 = Dummy()
    assert dummy_1 is dummy_2
    assert dummy_1 == dummy_2
