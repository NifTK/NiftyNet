# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import argparse

import tensorflow as tf

from niftynet.utilities.user_parameters_default import *
from tests.niftynet_testcase import NiftyNetTestCase

class TestUserParameters(NiftyNetTestCase):
    def test_list_all(self):
        test_parser = argparse.ArgumentParser(conflict_handler='resolve')
        test_parser = add_application_args(test_parser)
        test_parser = add_network_args(test_parser)
        test_parser = add_training_args(test_parser)
        test_parser = add_input_data_args(test_parser)
        test_parser = add_inference_args(test_parser)

        for opt in test_parser._actions:
            print(opt_to_string(opt))


def opt_to_string(opt):
    summary = 'opt: [{}]\n'.format(opt.dest)
    summary += '---- type: {}\n'.format(opt.type)
    summary += '---- default: {}\n'.format(opt.default)
    summary += '---- description: {}\n'.format(opt.help)
    return summary


if __name__ == "__main__":
    tf.test.main()
