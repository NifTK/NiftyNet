import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages


class BaseLayer(object):
    def __init__(self, is_training, device_str):
        self._device_string = device_str  # initialise on CPU by default
        self._is_training = is_training

        self.batch_size = 0
        self.num_classes = 0
        self.input_image_size = 0
        self.input_label_size = 0
        self.activation_type = ""

    def __init_variable(self, name, shape, init, trainable=True):
        with tf.device('/%s:0' % self._device_string):
            var = tf.get_variable(  # init variable if not exists
                name, shape, initializer=init, trainable=trainable)
            if trainable:
                tf.add_to_collection('reg_var', var)
        return var

    def __variable_with_weight_decay(self, name, shape, stddev):
        if name == 'b':  # default bias initialised to 0
            return self.__init_variable(
                name, shape,
                tf.constant_initializer(0.0, dtype=tf.float32),
                trainable=True)
        elif (name == 'w') and (stddev < 0):  # default weights initialiser
            stddev = np.sqrt(1.3 * 2.0 / (np.prod(shape[:-2]) * shape[-1]))
            return self.__init_variable(
                name, shape,
                tf.truncated_normal_initializer(
                    mean=0.0, stddev=stddev, dtype=tf.float32),
                trainable=True)
        elif name == 'w':  # initialiser with custom stddevs
            return self.__init_variable(
                name, shape,
                tf.truncated_normal_initializer(
                    mean=0.0, stddev=stddev, dtype=tf.float32),
                trainable=True)
        return None

    def batch_norm(self, inputs):
        x_shape = inputs.get_shape()
        params_shape = x_shape[-1:]
        axis = list(range(x_shape.ndims - 1))
        beta = self.__init_variable(
            'beta', params_shape,
            tf.constant_initializer(0.0, dtype=tf.float32), trainable=True)
        gamma = self.__init_variable(
            'gamma', params_shape,
            tf.constant_initializer(1.0, dtype=tf.float32), trainable=True)
        moving_mean = self.__init_variable(
            'moving_mean', params_shape,
            tf.constant_initializer(0.0, dtype=tf.float32), trainable=False)
        moving_variance = self.__init_variable(
            'moving_variance', params_shape,
            tf.constant_initializer(1.0, dtype=tf.float32), trainable=False)
        # mean and var
        counts = 1
        for d in axis:
            counts = counts * x_shape[d].value
        divisors = tf.constant(1.0 / counts, dtype=inputs.dtype)
        mean = tf.reduce_sum(inputs, axis) * divisors
        ## variance = sum((x-mean)^2)/n - (sum(x-mean)/n)^2
        variance = tf.subtract(
            tf.reduce_sum(tf.squared_difference(inputs, mean), axis) * divisors,
            tf.square(tf.reduce_sum(tf.subtract(inputs, mean), axis) * divisors))
        update_moving_mean = moving_averages.assign_moving_average(
            moving_mean, mean, 0.9)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, 0.9)

        if self._is_training:
            with tf.control_dependencies(
                    [update_moving_mean, update_moving_variance]):
                inputs = tf.nn.batch_normalization(
                    inputs, mean, variance, beta, gamma, 1e-5)
        else:
            inputs = tf.nn.batch_normalization(
                inputs, moving_mean, moving_variance, beta, gamma, 1e-5)
        return inputs

    def conv_1x1(self, f_in, ni_, no_):
        kernel = self.__variable_with_weight_decay(
            'w', [1, 1, 1, ni_, no_], -1)
        conv = tf.nn.conv3d(f_in, kernel, [1, 1, 1, 1, 1], padding='SAME')
        return conv

    def conv_3x3(self, f_in, ni_, no_):
        kernel = self.__variable_with_weight_decay(
            'w', [3, 3, 3, ni_, no_], -1)
        conv = tf.nn.conv3d(f_in, kernel, [1, 1, 1, 1, 1], padding='SAME')
        return conv

    def conv_layer_1x1(self, f_in, ni_, no_, bn=True, acti=True):
        kernel = self.__variable_with_weight_decay(
            'w', [1, 1, 1, ni_, no_], -1)
        biases = self.__variable_with_weight_decay(
            'b', [no_], 0.0)
        conv = tf.nn.conv3d(f_in, kernel, [1, 1, 1, 1, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        if bn:
            conv = self.batch_norm(conv)
        if acti:
            conv = self.nonlinear_acti(conv)
        return conv

    def set_activation_type(self, type_str):
        self.activation_type = type_str

    def nonlinear_acti(self, f_in):
        if self.activation_type == 'relu':
            return tf.nn.relu(f_in)

    @staticmethod
    def _print_activations(tf_var):
        print(tf_var.op.name, tf_var.get_shape().as_list()),
