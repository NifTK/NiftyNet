# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


# Before doing anything else, check TF is installed
# and fail gracefully if not.
try:
    import tensorflow as tf
except ImportError:
    raise ImportError('NiftyNet is based on TensorFlow, which'
                      ' does not seem to be installed on your'
                      ' system.\nPlease install TensorFlow'
                      ' (https://www.tensorflow.org/) to be'
                      ' able to use NiftyNet.')

import os
import sys
import logging as log

import niftynet.utilities.util_common as util
import niftynet.utilities.user_parameters_parser as user_parameters_parser
from niftynet.engine.application_driver import ApplicationDriver
from niftynet.io.misc_io import touch_folder

CONSOLE_LOG_FORMAT = '%(levelname)s:niftynet: %(message)s'
FILE_LOG_FORMAT = '%(levelname)s:niftynet:%(asctime)s: %(message)s'


def main():
    system_param, input_data_param = user_parameters_parser.run()
    if util.has_bad_inputs(system_param):
        return -1

    # print all parameters to txt file for future reference
    all_param = {}
    all_param.update(system_param)
    all_param.update(input_data_param)
    txt_file = 'settings_{}.txt'.format(system_param['APPLICATION'].action)
    model_folder = touch_folder(system_param['APPLICATION'].model_dir)
    txt_file = os.path.join(model_folder, txt_file)
    util.print_save_input_parameters(all_param, txt_file)

    # keep all commandline outputs
    log_file_name = os.path.join(
        model_folder,
        '{}_{}'.format(all_param['APPLICATION'].action, 'niftynet_log'))
    __set_logger(file_name=log_file_name)

    # start application
    app_driver = ApplicationDriver()
    app_driver.initialise_application(system_param, input_data_param)
    app_driver.run_application()
    return 0


def __set_logger(file_name=None):
    tf.logging._logger.handlers = []
    tf.logging._logger = log.getLogger('tensorflow')
    tf.logging.set_verbosity(tf.logging.INFO)

    f = log.Formatter(CONSOLE_LOG_FORMAT)
    std_handler = log.StreamHandler(sys.stdout)
    std_handler.setFormatter(f)
    tf.logging._logger.addHandler(std_handler)

    if file_name is not None:
        f = log.Formatter(FILE_LOG_FORMAT)
        file_handler = log.FileHandler(file_name)
        file_handler.setFormatter(f)
        tf.logging._logger.addHandler(file_handler)
