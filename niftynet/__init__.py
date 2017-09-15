# -*- coding: utf-8 -*-
"""

.. module:: niftynet
   :synopsis: Entry points for the NiftyNet CLI.

"""
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

import niftynet.utilities.util_common as util
import niftynet.utilities.user_parameters_parser as user_parameters_parser
from niftynet.engine.application_driver import ApplicationDriver
from niftynet.io.misc_io import touch_folder
from niftynet.io.misc_io import set_logger
from niftynet.io.misc_io import resolve_module_dir
from niftynet.io.misc_io import to_absolute_path


def main():
    system_param, input_data_param = user_parameters_parser.run()
    if util.has_bad_inputs(system_param):
        return -1

    # print all parameters to txt file for future reference
    all_param = {}
    all_param.update(system_param)
    all_param.update(input_data_param)

    # Set up path for niftynet model_root
    # (rewriting user input with an absolute path)
    system_param['SYSTEM'].model_dir = resolve_module_dir(
        system_param['SYSTEM'].model_dir,
        create_new=system_param['SYSTEM'].action == "train")

    # writing all params for future reference
    txt_file = 'settings_{}.txt'.format(system_param['SYSTEM'].action)
    txt_file = os.path.join(system_param['SYSTEM'].model_dir, txt_file)
    util.print_save_input_parameters(all_param, txt_file)

    # keep all commandline outputs to model_root
    log_file_name = os.path.join(
        system_param['SYSTEM'].model_dir,
        '{}_{}'.format(all_param['SYSTEM'].action, 'niftynet_log'))
    set_logger(file_name=log_file_name)

    # set up all model folder related parameters here
    # see https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/issues/168
    # 1. resolve mapping file:
    try:
        if system_param['NETWORK'].histogram_ref_file:
            system_param['NETWORK'].histogram_ref_file = to_absolute_path(
                input_path=system_param['NETWORK'].histogram_ref_file,
                model_root=system_param['SYSTEM'].model_dir)
    except (AttributeError, KeyError):
        pass
    # 2. resolve output file:
    try:
        if system_param['INFERENCE'].save_seg_dir:
            system_param['INFERENCE'].save_seg_dir = to_absolute_path(
                input_path=system_param['INFERENCE'].save_seg_dir,
                model_root=system_param['SYSTEM'].model_dir)
    except (AttributeError, KeyError):
        pass

    # start application
    app_driver = ApplicationDriver()
    app_driver.initialise_application(system_param, input_data_param)
    app_driver.run_application()
    return 0


