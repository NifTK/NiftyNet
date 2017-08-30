# -*- coding: utf-8 -*-

"""
To customise optimisers including
new optimisation methods, learning rate decay schedule,
or customise other optional parameters of the optimiser:

create a `newclass.py` that has a class `NewOptimisor` and implement
`get_instance()`.
and set config parameter in config file or from command line
specify `--optimiser newclass.NewOptimisor`
"""

from __future__ import absolute_import, print_function

import tensorflow as tf


class Adam(object):
    @staticmethod
    def get_instance(learning_rate):
        return tf.train.AdamOptimizer(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-08,
            use_locking=False, name='Adam')


class Adagrad(object):
    @staticmethod
    def get_instance(learning_rate):
        return tf.train.AdagradOptimizer(
            learning_rate=learning_rate,
            initial_accumulator_value=0.1,
            use_locking=False, name='Adagrad')


class Momentum(object):
    @staticmethod
    def get_instance(learning_rate):
        return tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9,
            use_locking=False,
            name='Momentum',
            use_nesterov=False)


class GradientDescent(object):
    @staticmethod
    def get_instance(learning_rate):
        return tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate,
            use_locking=False,
            name='GradientDescent')
