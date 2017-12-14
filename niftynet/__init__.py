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
                      ' system.\n\nPlease install TensorFlow'
                      ' (https://www.tensorflow.org/) to be'
                      ' able to use NiftyNet.')

try:
    minimal_required_version = "1.3"
    tf_version = tf.__version__
    if tf_version < minimal_required_version:
        tf.logging.fatal('TensorFlow %s or later is required.'
                         '\n\nPlease upgrade TensorFlow'
                         ' (https://www.tensorflow.org/) to be'
                         ' able to use NiftyNet.\nCurrently using '
                         'TensorFlow %s:\ninstalled at %s\n\n',
                         minimal_required_version, tf_version, tf.__file__)
        raise ImportError
    else:
        tf.logging.info('TensorFlow version %s', tf_version)
except AttributeError:
    pass

import os

import niftynet.utilities.util_common as util
import niftynet.utilities.user_parameters_parser as user_parameters_parser
from niftynet.engine.application_driver import ApplicationDriver
from niftynet.io.misc_io import touch_folder
from niftynet.io.misc_io import set_logger
from niftynet.io.misc_io import resolve_module_dir
from niftynet.io.misc_io import to_absolute_path


def main():
    set_logger()
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
    try:
        util.print_save_input_parameters(all_param, txt_file)
    except IOError:
        tf.logging.fatal(
            'Unable to write %s,\nplease check '
            'model_dir parameter, current value: %s',
            txt_file, system_param['SYSTEM'].model_dir)
        raise

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
    # 3. resolve dataset splitting file:
    try:
        if system_param['SYSTEM'].dataset_split_file:
            system_param['SYSTEM'].dataset_split_file = to_absolute_path(
                input_path=system_param['SYSTEM'].dataset_split_file,
                model_root=system_param['SYSTEM'].model_dir)
    except (AttributeError, KeyError):
        pass

    # start application
    app_driver = ApplicationDriver()
    app_driver.initialise_application(system_param, input_data_param)
    app_driver.run_application()
    return 0
