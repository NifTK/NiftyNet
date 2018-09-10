# -*- coding: utf-8 -*-
"""
Loading modules from a string representing the class name
or a short name that matches the dictionary item defined
in this module

all classes and docs are taken from
https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/python/ops/init_ops.py
"""
import tensorflow as tf

SEED = 42


class Constant(object):
    """
    initialize with a constant value
    """

    @staticmethod
    def get_instance(args):
        """
        create an instance of the initializer
        """
        value = float(args.get('value', 0.0))
        return tf.constant_initializer(value)


class Zeros(object):
    """
    initialize with zeros
    """

    @staticmethod
    def get_instance(args):
        # pylint: disable=unused-argument
        """
        create an instance of the initializer
        """
        return tf.constant_initializer(0.0)


class Ones(object):
    """
    initialize with ones
    """

    @staticmethod
    def get_instance(args):
        # pylint: disable=unused-argument
        """
        create an instance of the initializer
        """
        return tf.constant_initializer(1.0)


class UniformUnitScaling(object):
    """
    see also:
        https://www.tensorflow.org/api_docs/python/tf/uniform_unit_scaling_initializer
    """

    @staticmethod
    def get_instance(args):
        """
        create an instance of the initializer
        """
        factor = float(args.get('factor', 1.0))
        return tf.uniform_unit_scaling_initializer(factor, seed=SEED)


class Orthogonal(object):
    """
    see also:
        https://www.tensorflow.org/api_docs/python/tf/orthogonal_initializer
    """

    @staticmethod
    def get_instance(args):
        """
        create an instance of the initializer
        """
        gain = float(args.get('gain', 1.0))
        return tf.orthogonal_initializer(gain, seed=SEED)


class VarianceScaling(object):
    """
    see also:
        https://www.tensorflow.org/api_docs/python/tf/variance_scaling_initializer
    """

    @staticmethod
    def get_instance(args):
        """
        create an instance of the initializer
        """
        scale = float(args.get('scale', 1.0))
        mode = args.get('mode', "fan_in")
        assert (mode in ["fan_in", "fan_out", "fan_avg"])
        distribution = args.get('distribution', "normal")
        assert (distribution in ["normal", "uniform"])
        return tf.variance_scaling_initializer(scale,
                                               mode,
                                               distribution,
                                               seed=SEED)


class GlorotNormal(object):
    """
    see also:
        https://www.tensorflow.org/api_docs/python/tf/glorot_normal_initializer
    """

    @staticmethod
    def get_instance(args):
        # pylint: disable=unused-argument
        """
        create an instance of the initializer
        """
        return tf.glorot_normal_initializer(seed=SEED)


class GlorotUniform(object):
    """
    see also:
        https://www.tensorflow.org/api_docs/python/tf/glorot_uniform_initializer
    """

    @staticmethod
    def get_instance(args):
        # pylint: disable=unused-argument
        """
        create an instance of the initializer
        """
        return tf.glorot_uniform_initializer(seed=SEED)


class HeUniform(object):
    """
    He uniform variance scaling initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where ``limit`` is ``sqrt(6 / fan_in)``
    where ``fan_in`` is the number of input units in the weight tensor.
    # Arguments
    seed: A Python integer. Used to seed the random generator.
    # Returns
    An initializer.
    # References
    He et al., https://arxiv.org/abs/1502.01852
    """

    @staticmethod
    def get_instance(args):
        # pylint: disable=unused-argument
        """
        create an instance of the initializer
        """
        if not args:
            args = {"scale": "2.", "mode": "fan_in", "distribution": "uniform"}
        return VarianceScaling.get_instance(args)


class HeNormal(object):
    """
    He normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with ``stddev = sqrt(2 / fan_in)``
    where ``fan_in`` is the number of input units in the weight tensor.
    # Arguments
    seed: A Python integer. Used to seed the random generator.
    # Returns
    An initializer.
    # References
    He et al., https://arxiv.org/abs/1502.01852
    """

    @staticmethod
    def get_instance(args):
        # pylint: disable=unused-argument
        """
        create an instance of the initializer
        """
        if not args:
            args = {"scale": "2.", "mode": "fan_in", "distribution": "normal"}
        return VarianceScaling.get_instance(args)
