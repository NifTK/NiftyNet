# -*- coding: utf-8 -*-
"""
This module defines a general procedure for running evaluations
Example usage:
    app_driver = EvaluationApplicationDriver()
    app_driver.initialise_application(system_param, input_data_param)
    app_driver.run_application()

system_param and input_data_param should be generated using:
niftynet.utilities.user_parameters_parser.run()
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import itertools

import pandas as pd
import tensorflow as tf

from niftynet.engine.application_factory import ApplicationFactory
from niftynet.io.misc_io import touch_folder
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner

FILE_PREFIX = 'model.ckpt'


class EvaluationApplicationDriver(object):
    """
    This class represents the application logic for evaluating a set of
    results inferred within NiftyNet (or externally generated)
    """
    def __init__(self):
        self.app = None
        self.model_dir = None
        self.summary_dir = None
        self.session_prefix = None

        self.outputs_collector = None
        self.gradients_collector = None

    def initialise_application(self, workflow_param, data_param):
        """
        This function receives all parameters from user config file,
        create an instance of application.
        :param workflow_param: a dictionary of user parameters,
        keys correspond to sections in the config file
        :param data_param: a dictionary of input image parameters,
        keys correspond to data properties to be used by image_reader
        :return:
        """
        try:
            system_param = workflow_param.get('SYSTEM', None)
            net_param = workflow_param.get('NETWORK', None)
            infer_param = workflow_param.get('INFERENCE', None)
            eval_param = workflow_param.get('EVALUATION', None)
            app_param = workflow_param.get('CUSTOM', None)
        except AttributeError:
            tf.logging.fatal('parameters should be dictionaries')
            raise
        self.num_threads = 1
        # self.num_threads = max(system_param.num_threads, 1)
        # self.num_gpus = system_param.num_gpus
        # set_cuda_device(system_param.cuda_devices)

        # set output TF model folders
        self.model_dir = touch_folder(
            os.path.join(system_param.model_dir, 'models'))
        self.session_prefix = os.path.join(self.model_dir, FILE_PREFIX)

        assert infer_param, 'inference parameters not specified'

        # create an application instance
        assert app_param, 'application specific param. not specified'
        self.app_param = app_param
        app_module = ApplicationFactory.create(app_param.name)
        self.app = app_module(net_param, infer_param, system_param.action)

        self.eval_param = eval_param

        data_param, self.app_param = \
            self.app.add_inferred_output(data_param, self.app_param)
        # initialise data input
        data_partitioner = ImageSetsPartitioner()
        # clear the cached file lists
        data_partitioner.reset()
        if data_param:
            data_partitioner.initialise(
                data_param=data_param,
                new_partition=False,
                ratios=None,
                data_split_file=system_param.dataset_split_file)

        # initialise data input
        self.app.initialise_dataset_loader(data_param, self.app_param,
                                           data_partitioner)
        self.app.initialise_evaluator(eval_param)

    def run(self, application):
        """
        This is the main application logic for evaluation.
        Computation of all metrics for all subjects is delegated to an
        Evaluator objects owned by the application object. The resulting
        metrics are aggregated as defined by the evaluation classes and
        output to one or more csv files (based on their 'group_by' headings).
        For example, per-subject metrics will be in one file, per-label-class
        metrics will be in another and per-subject-per-class will be in a
        third.
        :return:
        """
        start_time = time.time()
        try:
            if not os.path.exists(self.eval_param.save_csv_dir):
                os.makedirs(self.eval_param.save_csv_dir)
            # iteratively run the graph
            all_results = application.evaluator.evaluate()
            for group_by, data_frame in all_results.items():
                if group_by == (None,):
                    csv_id = ''
                else:
                    csv_id = '_'.join(group_by)

                with open(os.path.join(self.eval_param.save_csv_dir,
                                       'eval_' + csv_id + '.csv'), 'w') as csv:
                    csv.write(data_frame.reset_index().to_csv(index=False))
        except KeyboardInterrupt:
            tf.logging.warning('User cancelled application')
        except RuntimeError:
            import sys
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=sys.stdout)
        finally:
            tf.logging.info('Cleaning up...')
            tf.logging.info(
                "%s stopped (time in second %.2f).",
                type(application).__name__, (time.time() - start_time))
