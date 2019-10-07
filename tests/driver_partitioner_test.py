# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import tensorflow as tf

from niftynet.engine.application_driver import ApplicationDriver
from niftynet.utilities.util_common import ParserNamespace
from tests.niftynet_testcase import NiftyNetTestCase

TARGET_FILE = os.path.join('testing_data', 'test_splitting.csv')


def _generate_base_params():
    # initialise compulsory params that are irrelevant
    # to this unit test
    user_param = dict()
    user_param['SYSTEM'] = ParserNamespace(
        model_dir='./testing_data',
        num_threads=2,
        num_gpus=1,
        cuda_devices='',
        event_handler=None,
        iteration_generator=None)

    user_param['NETWORK'] = ParserNamespace(
        batch_size=20,
        name='tests.toy_application.TinyNet')

    user_param['TRAINING'] = ParserNamespace(
        starting_iter=0,
        max_iter=2,
        save_every_n=2,
        tensorboard_every_n=0,
        max_checkpoints=100)

    user_param['INFERENCE'] = ParserNamespace(
        inference_iter=-1)

    user_param['CUSTOM'] = ParserNamespace(
        name='tests.toy_application.ToyApplication',
        vector_size=100,
        mean=10.0,
        stddev=2.0)
    return user_param


def _generate_data_param():
    user_param = dict()
    user_param['modality'] = ParserNamespace(
        csv_file=os.path.join('testing_data', 'mod1test.csv'),
        path_to_search='testing_data',
        filename_contains='nii')

    user_param['modality2'] = ParserNamespace(
        csv_file=os.path.join('testing_data', 'mod2test.csv'),
        path_to_search='testing_data',
        filename_contains='nii')
    return user_param


def generate_input_params(**arg_dicts):
    user_param = _generate_base_params()
    for key in list(arg_dicts):
        if not arg_dicts[key]:
            continue
        user_param[key].update(**arg_dicts[key])
    return user_param


def clear_target():
    if not os.path.isfile(TARGET_FILE):
        return
    os.remove(TARGET_FILE)


def write_target():
    clear_target()
    user_param = generate_input_params(
        SYSTEM={'action': 'train',
                'dataset_split_file': TARGET_FILE
                },
        TRAINING={'validation_every_n': 2,
                  'validation_max_iter': 1,
                  'exclude_fraction_for_validation': 0.1,
                  'exclude_fraction_for_inference': 0.1,
                  }
    )
    data_param = _generate_data_param()
    app_driver = ApplicationDriver()
    app_driver.initialise_application(user_param, data_param)
    assert os.path.isfile(TARGET_FILE)
    return


class DriverPartitionerTestExistingFile(NiftyNetTestCase):
    def test_training(self):
        write_target()
        user_param = generate_input_params(
            SYSTEM={'action': 'train',
                    'dataset_split_file': TARGET_FILE
                    },
            TRAINING={'validation_every_n': 2,
                      'validation_max_iter': 1,
                      'exclude_fraction_for_validation': 0.1,
                      'exclude_fraction_for_inference': 0.1,
                      }
        )
        data_param = _generate_data_param()
        app_driver = ApplicationDriver()
        app_driver.initialise_application(user_param, data_param)
        partitioner = app_driver.data_partitioner
        self.assertTrue(partitioner.has_training)
        self.assertTrue(partitioner.has_inference)
        self.assertTrue(partitioner.has_validation)

    def test_training_no_validation(self):
        write_target()
        user_param = generate_input_params(
            SYSTEM={'action': 'train',
                    'dataset_split_file': TARGET_FILE
                    },
            TRAINING={'validation_every_n': -1,
                      'validation_max_iter': 1,
                      'exclude_fraction_for_validation': 0.0,
                      'exclude_fraction_for_inference': 0.0,
                      }
        )
        data_param = _generate_data_param()
        app_driver = ApplicationDriver()
        app_driver.initialise_application(user_param, data_param)
        partitioner = app_driver.data_partitioner
        self.assertTrue(partitioner.has_training)
        self.assertTrue(partitioner.has_inference)
        self.assertTrue(partitioner.has_validation)

    def test_inference_no_validation(self):
        write_target()
        user_param = generate_input_params(
            SYSTEM={'action': 'inference',
                    'dataset_split_file': TARGET_FILE
                    },
            TRAINING={'validation_every_n': -1,
                      'validation_max_iter': 1,
                      'exclude_fraction_for_validation': 0.0,
                      'exclude_fraction_for_inference': 0.0,
                      }
        )
        data_param = _generate_data_param()
        app_driver = ApplicationDriver()
        app_driver.initialise_application(user_param, data_param)
        partitioner = app_driver.data_partitioner
        self.assertTrue(partitioner.has_training)
        self.assertTrue(partitioner.has_inference)
        self.assertTrue(partitioner.has_validation)

    def test_inference_validation(self):
        write_target()
        user_param = generate_input_params(
            SYSTEM={'action': 'inference',
                    'dataset_split_file': TARGET_FILE
                    },
            TRAINING={'validation_every_n': 10,
                      'validation_max_iter': 1,
                      'exclude_fraction_for_validation': 0.0,
                      'exclude_fraction_for_inference': 0.0,
                      }
        )
        data_param = _generate_data_param()
        app_driver = ApplicationDriver()
        app_driver.initialise_application(user_param, data_param)
        partitioner = app_driver.data_partitioner
        self.assertTrue(partitioner.has_training)
        self.assertTrue(partitioner.has_inference)
        self.assertTrue(partitioner.has_validation)


class DriverPartitionerTestNoFile(NiftyNetTestCase):
    def test_training(self):
        clear_target()
        user_param = generate_input_params(
            SYSTEM={'action': 'train',
                    'dataset_split_file': TARGET_FILE
                    },
            TRAINING={'validation_every_n': 2,
                      'validation_max_iter': 1,
                      'exclude_fraction_for_validation': 0.1,
                      'exclude_fraction_for_inference': 0.1,
                      }
        )
        data_param = _generate_data_param()
        app_driver = ApplicationDriver()
        app_driver.initialise_application(user_param, data_param)
        partitioner = app_driver.data_partitioner
        self.assertTrue(partitioner.has_training)
        self.assertTrue(partitioner.has_inference)
        self.assertTrue(partitioner.has_validation)

    def test_training_no_validation(self):
        clear_target()
        user_param = generate_input_params(
            SYSTEM={'action': 'train',
                    'dataset_split_file': TARGET_FILE
                    },
            TRAINING={'validation_every_n': -1,
                      'validation_max_iter': 1,
                      'exclude_fraction_for_validation': 0.0,
                      'exclude_fraction_for_inference': 0.0,
                      }
        )
        data_param = _generate_data_param()
        app_driver = ApplicationDriver()
        app_driver.initialise_application(user_param, data_param)
        partitioner = app_driver.data_partitioner
        self.assertFalse(partitioner.has_training)
        self.assertFalse(partitioner.has_inference)
        self.assertFalse(partitioner.has_validation)
        self.assertTrue(partitioner.all_files is not None)

        clear_target()
        user_param = generate_input_params(
            SYSTEM={'action': 'train',
                    'dataset_split_file': TARGET_FILE
                    },
            TRAINING={'validation_every_n': -1,
                      'validation_max_iter': 1,
                      'exclude_fraction_for_validation': 0.0,
                      'exclude_fraction_for_inference': 0.0,
                      }
        )
        data_param = _generate_data_param()
        app_driver = ApplicationDriver()
        app_driver.initialise_application(user_param, data_param)
        partitioner = app_driver.data_partitioner
        self.assertFalse(partitioner.has_training)
        self.assertFalse(partitioner.has_inference)
        self.assertFalse(partitioner.has_validation)
        self.assertTrue(partitioner.all_files is not None)

    def test_inference(self):
        clear_target()
        user_param = generate_input_params(
            SYSTEM={'action': 'inference',
                    'dataset_split_file': TARGET_FILE
                    },
            TRAINING={'validation_every_n': 1,
                      'validation_max_iter': 1,
                      'exclude_fraction_for_validation': 0.1,
                      'exclude_fraction_for_inference': 0.0,
                      }
        )
        data_param = _generate_data_param()
        app_driver = ApplicationDriver()
        app_driver.initialise_application(user_param, data_param)

        self.assertTrue(app_driver.data_partitioner is not None)
        self.assertFalse(os.path.isfile(TARGET_FILE))
        partitioner = app_driver.data_partitioner
        self.assertTrue(partitioner._partition_ids is None)

    def test_inference_no_validation(self):
        clear_target()
        user_param = generate_input_params(
            SYSTEM={'action': 'inference',
                    'dataset_split_file': TARGET_FILE
                    },
        )
        data_param = _generate_data_param()
        app_driver = ApplicationDriver()
        app_driver.initialise_application(user_param, data_param)

        self.assertTrue(app_driver.data_partitioner is not None)
        self.assertFalse(os.path.isfile(TARGET_FILE))
        partitioner = app_driver.data_partitioner
        self.assertTrue(partitioner._partition_ids is None)


class DriverPartitionerTestNoData(NiftyNetTestCase):
    def test_no_data_param_infer(self):
        clear_target()
        user_param = generate_input_params(
            SYSTEM={'action': 'inference',
                    'dataset_split_file': TARGET_FILE
                    }
        )
        app_driver = ApplicationDriver()
        app_driver.initialise_application(user_param, {})
        self.assertTrue(app_driver.data_partitioner is not None)
        self.assertFalse(os.path.isfile(TARGET_FILE))

        partitioner = app_driver.data_partitioner
        self.assertFalse(partitioner.all_files)

    def test_no_data_param_train(self):
        clear_target()
        user_param = generate_input_params(
            SYSTEM={'action': 'train',
                    'dataset_split_file': TARGET_FILE
                    },
            TRAINING={'validation_every_n': -1,
                      'exclude_fraction_for_validation': 0.1,
                      'exclude_fraction_for_inference': 0.1,
                      }
        )
        app_driver = ApplicationDriver()
        app_driver.initialise_application(user_param, {})
        self.assertTrue(app_driver.data_partitioner is not None)
        self.assertFalse(os.path.isfile(TARGET_FILE))

        partitioner = app_driver.data_partitioner
        self.assertFalse(partitioner.all_files)


if __name__ == "__main__":
    tf.test.main()
