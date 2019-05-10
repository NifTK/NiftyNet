# -*- coding: utf-8 -*-
"""
This module contains the NiftyNet application module wrapper and
specialisations for NiftyNet's standard applications.
"""
from __future__ import absolute_import

from niftynet.engine.application_driver import ApplicationDriver
from niftynet.engine.signal import INFER, TRAIN, EVAL
from niftynet.evaluation.evaluation_application_driver import \
    EvaluationApplicationDriver
from niftynet.io.misc_io import resolve_module_dir, to_absolute_path
from niftynet.io.memory_image_sets_partitioner import \
    MEMORY_INPUT_NUM_SUBJECTS_PARAM
from niftynet.io.memory_image_sink import MEMORY_OUTPUT_CALLBACK_PARAM
from niftynet.io.memory_image_source import make_input_spec
from niftynet.utilities.user_parameters_parser import extract_app_parameters,\
    ACTIONS

SUPPORTED_ACTIONS = tuple(ACTIONS.keys())


class ApplicationModuleWrapper(object):
    """
    This class enables the use of NiftyNet applications as
    standard class instances in 3rd party code.
    """

    def __init__(self,
                 app_name,
                 model_file):
        """
        :param model_file: a NiftyNet application configuration file
        :param app_name: name of registered NiftyNet application
        """

        self._model_file = model_file
        self._app = app_name
        self._output_callback = None
        self._num_subjects = 0
        self._input_callbacks = {}
        self._action = None

    def set_output_callback(self, output_callback):
        """
        Sets the callback function receiving the output images
        :param output_callback: a function accepting an image and a subject ID
        :return: self
        """

        self._output_callback = output_callback

        return self

    def set_num_subjects(self, num_subjects):
        """
        Sets the number of subjects expected to be found in memory
        :param num_subjects: a positive integer number of subjects.
        :return: self
        """

        self._num_subjects = num_subjects

        return self

    def set_input_callback(self, modality, input_callback):
        """
        Sets the callback function for the input images
        :param modality: name of a modality from the model config file
        :param input_callback: a function receiving an index and yielding
        an image tensor.
        :return: self
        """

        self._input_callbacks[modality] = input_callback

        return self

    def set_action(self, action):
        """
        Sets the application action to perform. TRAIN, INFER, etc.
        See also niftynet.engine.signal
        :param action: one of SUPPORTED_ACTIONS
        :return: self
        """

        for ref in SUPPORTED_ACTIONS:
            if ref.startswith(action.lower()):
                action = ref
                break

        self._action = action

        return self

    def _install_image_callbacks(self, data_param, infer_param):
        """
        Makes the I/O callbacks visible to the app driver and other
        application admin logic.
        :param data_param: data specification
        :param action_param: application specific settings
        :return: the updated data_param and app_param
        """

        for name, funct in self._input_callbacks.items():
            if not name in data_param:
                raise RuntimeError('Have an input callback function'
                                   ' %s without corresponding data '
                                   'specification section. Found sections:'
                                   ' %s' % (name, list(data_param.keys())))

            make_input_spec(data_param[name], funct)

        if self._output_callback and not infer_param is None:
            vars(infer_param)[MEMORY_OUTPUT_CALLBACK_PARAM] \
                = self._output_callback

        data_param[MEMORY_INPUT_NUM_SUBJECTS_PARAM] = self._num_subjects

    def _check_configured(self):
        """
        Checks if all required image I/O settings have been
        specified. Throws a RuntimeException, otherwise.
        """

        if self._num_subjects <= 0:
            raise RuntimeError('A positive number of subjects must be set')

        if not self._action \
           or self._action not in SUPPORTED_ACTIONS:
            raise RuntimeError('A supported action must be set (%s); got: %s'
                               % (SUPPORTED_ACTIONS, self._action))

        if not self._input_callbacks:
            raise RuntimeError('Input image callbacks must be set.')

        if INFER.startswith(self._action) \
                and not self._output_callback:
            raise RuntimeError('For evaluations, an output callback function'
                               ' must be set')

    def initialise_application(self):
        """
        Loads and configures the application encapsulated by this
        object for the given action.
        :return: self
        """

        self._check_configured()

        system_param, input_data_param = extract_app_parameters(
            self._app, self._model_file, self._action)

        resolve_module_dir(system_param['SYSTEM'].model_dir,
                           create_new=TRAIN.startswith(self._action))

        # Verbatim copy of F/S setup from NiftyNet's main
        try:
            if system_param['NETWORK'].histogram_ref_file:
                system_param['NETWORK'].histogram_ref_file = to_absolute_path(
                    input_path=system_param['NETWORK'].histogram_ref_file,
                    model_root=system_param['SYSTEM'].model_dir)
        except (AttributeError, KeyError):
            pass

        try:
            if system_param['INFERENCE'].save_seg_dir:
                system_param['INFERENCE'].save_seg_dir = to_absolute_path(
                    input_path=system_param['INFERENCE'].save_seg_dir,
                    model_root=system_param['SYSTEM'].model_dir)
        except (AttributeError, KeyError):
            pass

        try:
            if system_param['SYSTEM'].dataset_split_file:
                system_param['SYSTEM'].dataset_split_file = to_absolute_path(
                    input_path=system_param['SYSTEM'].dataset_split_file,
                    model_root=system_param['SYSTEM'].model_dir)
        except (AttributeError, KeyError):
            pass

        try:
            if system_param['EVALUATION'].save_csv_dir:
                system_param['EVALUATION'].save_csv_dir = to_absolute_path(
                    input_path=system_param['EVALUATION'].save_csv_dir,
                    model_root=system_param['SYSTEM'].model_dir)
        except (AttributeError, KeyError):
            pass

        driver_table = {
            TRAIN: ApplicationDriver,
            INFER: ApplicationDriver,
            EVAL: EvaluationApplicationDriver
        }

        infer_param = system_param.get('INFERENCE', None)
        self._install_image_callbacks(input_data_param, infer_param)

        app_driver = driver_table[ACTIONS[self._action]]()
        app_driver.initialise_application(system_param, input_data_param)

        self._app = app_driver

        # Have to remove number of subjects from data_param since
        # otherwise it gets interpretted as a modality
        del input_data_param[MEMORY_INPUT_NUM_SUBJECTS_PARAM]

        return self

    def run(self):
        """
        Runs the application main loop
        """

        if isinstance(self._app, str):
            raise RuntimeError('The module must be initialised before it'
                               ' can be run.')

        self._app.run(self._app.app)


def _make_wrapper(app_class, app_name):
    """
    Creates a module wrapper class for a specified application class.
    :param app_class: application class, e.g., SegmentationApplication
    :param app_name: front-end name, e.g, net_regress
    """

    class _ClassModule(ApplicationModuleWrapper):
        """
        {} as a module
        """

        def __init__(self, model_file):
            """
            Instantiates {} as a module
            :param model_file: model and data specification file
            """

            super(_ClassModule, self).__init__(app_name,
                                               model_file)

    cls = _ClassModule

    cls.__doc__ = cls.__doc__.format(str(app_class))
    cls.__init__.__doc__ = cls.__init__.__doc__.format(str(app_class))

    return cls


SegmentationApplicationModule \
    = _make_wrapper('SegmentationApplication', 'net_segment')

RegressionApplicationModule \
    = _make_wrapper('RegressionApplication', 'net_regress')

AutoencoderApplicationModule \
    = _make_wrapper('AutoencoderApplication', 'net_autoencoder')

GANApplicationModule \
    = _make_wrapper('GANApplication', 'net_gan')

ClassificationApplicationModule \
    = _make_wrapper('ClassificationApplication', 'net_classify')
