# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

# Before doing anything else, check TF is installed
# and fail gracefully if not.
try:
    import tensorflow
except ImportError:
    raise ImportError('NiftyNet is based on TensorFlow, which'
                      ' does not seem to be installed on your'
                      ' system.\nPlease install TensorFlow'
                      ' (https://www.tensorflow.org/) to be'
                      ' able to use NiftyNet.')

import os

import niftynet.utilities.util_common as util
import niftynet.utilities.user_parameters_parser as user_parameters_parser

from niftynet.engine.application_driver import ApplicationDriver


def main():
    system_param, input_data_param = user_parameters_parser.run()
    if util.has_bad_inputs(system_param):
        return -1

    # print all parameters to txt file for future reference
    all_param = {}
    all_param.update(system_param)
    all_param.update(input_data_param)
    txt_file = 'settings_{}.txt'.format(system_param['APPLICATION'].action)
    txt_file = os.path.join(system_param['APPLICATION'].model_dir, txt_file)
    util.print_save_input_parameters(all_param, txt_file)

    app_driver = ApplicationDriver()
    app_driver.initialise_application(system_param, input_data_param)
    app_driver.run_application()
    return 0
