"""
This module contains the NiftyNet application module wrapper and
associated utilities
"""
from niftynet.engine.application_driver import ApplicationDriver, \
    EvaluationApplicationDriver
from niftynet.engine.signal import TRAIN, INFER, EVAL
from niftynet.io.misc_io import resolve_module_dir, to_absolute_path
from niftynet.utilities.user_parameters_parser import extract_app_parameters

class ApplicationModuleWrapper(object):
    """
    This class enables the use of NiftyNet applications as
    standard class instances in 3rd party code.
    """

    def __init__(self,
                 model_file,
                 app_name):
        """
        :param model_file: a NiftyNet application configuration file
        :param app_name: name of registered NiftyNet application
        """

        self._model_file = model_file
        self._app = app_name
        self._output_callback = None
        self._input_callbacks = {}

    def set_output_callback(self, output_callback):
        """
        Sets the callback function receiving the output images
        :param output_callback: a function accepting an image and a subject ID
        """

        self._output_callback = output_callback

    def set_num_subjects(self, num_subjects):
        """
        Sets the number of subjects expected to be found in memory
        :param num_subjects: a positive integer number of subjects.
        """

    def set_input_callback(self, modality, input_callback):
        """
        Sets the callback function for the input images
        :param modality: name of a modality from the model config file
        :param input_callback: a function receiving an index and yielding
        an image tensor.
        """

        self._input_callbacks[modality] = input_callback

    def initialise_application(self, action):
        """
        Loads and configures the application encapsulated by this
        object for the given action.
        :param action: a NiftyNet action from niftynet.engine.signal
        """

        system_param, input_data_param = extract_app_parameters(
            self._app, self._model_file, action)

        resolve_module_dir(system_param['SYSTEM'].model_dir,
                           create_new=action == TRAIN)

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
            EVAL: EvaluationApplicationDriver}

#        system_param['NETWORK'].output_callback = self._output_callback
#        system_param['NETWORK'].input_callback = self._input_callbacks

        app_driver = driver_table[system_param['SYSTEM'].action]()
        app_driver.initialise_application(system_param, input_data_param)
        app_driver.run(app_driver.app)

        self._app = app_driver
