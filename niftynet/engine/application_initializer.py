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
    """Initializer that generates tensors without scaling variance.
    When initializing a deep network, it is in principle advantageous to keep
    the scale of the input variance constant, so it does not explode or diminish
    by reaching the final layer. If the input is `x` and the operation `x * W`,
    and we want to initialize `W` uniformly at random, we need to pick `W` from
      [-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]
    to keep the scale intact, where `dim = W.shape[0]` (the size of the input).
    A similar calculation for convolutional networks gives an analogous result
    with `dim` equal to the product of the first 3 dimensions.  When
    nonlinearities are present, we need to multiply this by a constant `factor`.
    See [Sussillo et al., 2014](https://arxiv.org/abs/1412.6558)
    ([pdf](http://arxiv.org/pdf/1412.6558.pdf)) for deeper motivation,
    experiments
    and the calculation of constants. In section 2.3 there, the constants were
    numerically computed: for a linear layer it's 1.0, relu: ~1.43, tanh: ~1.15.
    Args:
    factor: Float.  A multiplicative factor by which the values will be scaled.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    dtype: The data type. Only floating point types are supported.
    """

    @staticmethod
    def get_instance(args):
        """
        create an instance of the initializer
        """
        factor = float(args.get('factor', 1.0))
        return tf.uniform_unit_scaling_initializer(factor, seed=SEED)


class Orthogonal(object):
    """Initializer that generates an orthogonal matrix.
    If the shape of the tensor to initialize is two-dimensional, i
    is initialized with an orthogonal matrix obtained from the singular value
    decomposition of a matrix of uniform random numbers.
    If the shape of the tensor to initialize is more than two-dimensional,
    a matrix of shape `(shape[0] * ... * shape[n - 2], shape[n - 1])`
    is initialized, where `n` is the length of the shape vector.
    The matrix is subsequently reshaped to give a tensor of the desired shape.
    Args:
    gain: multiplicative factor to apply to the orthogonal matrix
    dtype: The type of the output.
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
      """

    @staticmethod
    def get_instance(args):
        """
        create an instance of the initializer
        """
        gain = float(args.get('gain', 1.0))
        return tf.orthogonal_initializer(gain, seed=SEED)


class VarianceScaling(object):
    """Initializer capable of adapting its scale to the shape of
    weights tensors.
    With `distribution="normal"`, samples are drawn from a truncated normal
    distribution centered on zero, with `stddev = sqrt(scale / n)`
    where n is:
    - number of input units in the weight tensor, if mode = "fan_in"
    - number of output units, if mode = "fan_out"
    - average of the numbers of input and output units, if mode = "fan_avg"
    With `distribution="uniform"`, samples are drawn from a uniform
    distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.
    Arguments:
    scale: Scaling factor (positive float).
    mode: One of "fan_in", "fan_out", "fan_avg".
    distribution: Random distribution to use. One of "normal", "uniform".
    seed: A Python integer. Used to create random seeds. See
     @{tf.set_random_seed}
     for behavior.
    dtype: The data type. Only floating point types are supported.
    Raises:
    ValueError: In case of an invalid value for the "scale", mode" or
     "distribution" arguments.
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
    """The Glorot normal initializer, also called Xavier normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    Reference: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    Arguments:
    seed: A Python integer. Used to create random seeds. See
    @{tf.set_random_seed}
    for behavior.
    dtype: The data type. Only floating point types are supported.
    Returns:
    An initializer.
    """

    @staticmethod
    def get_instance(args):
        # pylint: disable=unused-argument
        """
        create an instance of the initializer
        """
        return tf.glorot_normal_initializer(seed=SEED)


class GlorotUniform(object):
    """The Glorot uniform initializer, also called Xavier uniform initializer.
    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.
    Reference: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
    Arguments:
    seed: A Python integer. Used to create random seeds. See
      @{tf.set_random_seed}
      for behavior.
    dtype: The data type. Only floating point types are supported.
    Returns:
    An initializer.
    """

    @staticmethod
    def get_instance(args):
        # pylint: disable=unused-argument
        """
        create an instance of the initializer
        """
        return tf.glorot_uniform_initializer(seed=SEED)


class HeUniform(object):
    """He uniform variance scaling initializer.
     It draws samples from a uniform distribution within [-limit, limit]
     where `limit` is `sqrt(6 / fan_in)`
     where `fan_in` is the number of input units in the weight tensor.
     # Arguments
         seed: A Python integer. Used to seed the random generator.
     # Returns
         An initializer.
     # References
         He et al., http://arxiv.org/abs/1502.01852
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
    """He normal initializer.
    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.
    # Arguments
        seed: A Python integer. Used to seed the random generator.
    # Returns
        An initializer.
    # References
        He et al., http://arxiv.org/abs/1502.01852
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
