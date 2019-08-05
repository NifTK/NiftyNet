from __future__ import absolute_import, print_function

import tensorflow as tf
import niftynet.engine.application_factory as Factory
from tests.niftynet_testcase import NiftyNetTestCase

class FactoryTest(NiftyNetTestCase):
    def test_import(self):
        var_names = [
            item for item in list(dir(Factory)) if item.startswith("SUPPORTED")]
        for var_name in var_names:
            mod_table = Factory.__dict__[var_name]
            for mod_name in list(mod_table):
                Factory.select_module(mod_name, 'test', mod_table)



if __name__ == "__main__":
    tf.test.main()
