import tensorflow as tf
from tensorflow.python.training import moving_averages

from .base_layer import TrainableLayer


def _compute_mean_and_var(inputs, axes):
    # compute mean and variance of the inputs along axes
    input_shape = inputs.get_shape()
    counts = 1
    for d in axes:
        counts = counts * input_shape[d].value
    divisors = tf.constant(1.0 / counts, dtype=inputs.dtype)
    mean = tf.reduce_sum(inputs, axes) * divisors
    # variance = sum((x-mean)^2)/n - (sum(x-mean)/n)^2
    variance = tf.subtract(
        tf.reduce_sum(tf.squared_difference(inputs, mean), axes) * divisors,
        tf.square(tf.reduce_sum(tf.subtract(inputs, mean), axes) * divisors))
    return mean, variance


class BNLayer(TrainableLayer):
    """
    Batch normalisation layer, with trainable mean value 'beta' and
    std 'gamma'.  'beta' is initialised to 0.0 and 'gamma' is initialised
    to 1.0.  This class assumes 'beta' and 'gamma' share the same type of
    regulariser.
    """

    def __init__(self,
                 regularizer=None,
                 moving_decay=0.9,
                 eps=1e-5,
                 name='batch_norm'):
        super(BNLayer, self).__init__(name=name)
        self.eps = eps
        self.moving_decay = moving_decay

        self.initializers = {'beta': tf.constant_initializer(0.0),
                             'gamma': tf.constant_initializer(1.0),
                             'moving_mean': tf.constant_initializer(0.0),
                             'moving_variance': tf.constant_initializer(1.0)}

        self.regularizers = {'beta': regularizer, 'gamma': regularizer}

    def layer_op(self, inputs, is_training, use_local_stats=False):
        input_shape = inputs.get_shape()

        # operates on all dims except the last dim
        params_shape = input_shape[-1:]
        axes = list(range(input_shape.ndims - 1))

        # create trainable variables and moving average variables
        beta = tf.get_variable(
            'beta',
            shape=params_shape,
            initializer=self.initializers['beta'],
            regularizer=self.regularizers['beta'],
            dtype=tf.float32, trainable=True)
        gamma = tf.get_variable(
            'gamma',
            shape=params_shape,
            initializer=self.initializers['gamma'],
            regularizer=self.regularizers['gamma'],
            dtype=tf.float32, trainable=True)
        moving_mean = tf.get_variable(
            'moving_mean',
            shape=params_shape,
            initializer=self.initializers['moving_mean'],
            dtype=tf.float32, trainable=False)
        moving_variance = tf.get_variable(
            'moving_variance',
            shape=params_shape,
            initializer=self.initializers['moving_variance'],
            dtype=tf.float32, trainable=False)

        # mean and var
        mean, variance = _compute_mean_and_var(inputs, axes)
        update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, self.moving_decay).op
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, self.moving_decay).op

        # call the normalisation function
        if is_training or use_local_stats:
            with tf.control_dependencies(
                    [update_moving_mean, update_moving_variance]):
                outputs = tf.nn.batch_normalization(
                    inputs, mean, variance,
                    beta, gamma, self.eps, name='batch_norm')
        else:
            outputs = tf.nn.batch_normalization(
                inputs, moving_mean, moving_variance,
                beta, gamma, self.eps, name='batch_norm')
        return outputs
