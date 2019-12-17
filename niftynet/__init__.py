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
    from distutils.version import LooseVersion

    minimal_required_version = LooseVersion("1.5")
    tf_version = LooseVersion(tf.__version__)
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

from niftynet.utilities.versioning import get_niftynet_version_string

__version__ = get_niftynet_version_string()

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

from niftynet.io.misc_io import set_logger, close_logger

set_logger()

from niftynet.utilities.util_import import require_module

require_module('blinker', descriptor='New dependency', mandatory=True)

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

from niftynet.engine.signal import TRAIN, INFER, EVAL
import niftynet.utilities.util_common as util
import niftynet.utilities.user_parameters_parser as user_parameters_parser
from niftynet.engine.application_driver import ApplicationDriver
from niftynet.evaluation.evaluation_application_driver import \
    EvaluationApplicationDriver
from niftynet.io.misc_io import touch_folder
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
        create_new=system_param['SYSTEM'].action == TRAIN)

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

    # 4. resolve evaluation dir:
    try:
        if system_param['EVALUATION'].save_csv_dir:
            system_param['EVALUATION'].save_csv_dir = to_absolute_path(
                input_path=system_param['EVALUATION'].save_csv_dir,
                model_root=system_param['SYSTEM'].model_dir)
    except (AttributeError, KeyError):
        pass

    # start application
    driver_table = {
        TRAIN: ApplicationDriver,
        INFER: ApplicationDriver,
        EVAL: EvaluationApplicationDriver}
    app_driver = driver_table[system_param['SYSTEM'].action]()
    app_driver.initialise_application(system_param, input_data_param)
    app_driver.run(app_driver.app)

    if tf.get_default_session() is not None:
        tf.get_default_session().close()
    tf.reset_default_graph()
    close_logger()

    return 0
