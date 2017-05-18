import numpy as np
import tensorflow as tf


class Layer(object):
    def __init__(self, name='untitled_op'):
        self._op = tf.make_template(name, self.layer_op, create_scope_now_=True)

    def layer_op(self, *args, **kwargs):
        msg = 'method \'layer_op\' in \'{}\''.format(type(self).__name__)
        raise NotImplementedError(msg)

    def __call__(self, *args, **kwargs):
        return self._op(*args, **kwargs)

    def __str__(self):
        return self.to_string()

    def layer_scope(self):
        return self._op.variable_scope

    def to_string(self):
        layer_scope_name = self.layer_scope().name
        out_str = "\033[42m[Layer]\033[0m {}".format(layer_scope_name)
        if not self._op._variables_created:
            out_str += ' \033[46m(input undecided)\033[0m'
            return out_str
        return out_str


class TrainableLayer(Layer):
    def __init__(self, name='untitled_op'):
        super(TrainableLayer, self).__init__(name=name)
        self._rand_initializer = None
        self._rand_regularizer = None
        self._const_initializer = None
        self._const_regularizer = None

    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 self.layer_scope().name)

    def num_trainable_params(self):
        n = tf.Dimension(0)
        for x in self.trainable_variables():
            n += np.prod(x.get_shape())
        return n

    def to_string(self):
        out_str = Layer.to_string(self)
        # try to add trainable variable info to the string
        layer_variables = self.trainable_variables()
        if not layer_variables:
            return out_str
        # including name of parameters
        out_str += ' \033[92m[Trainable]\033[0m '
        out_str += ', '.join(
            [v.name.split(':')[0][len(self.layer_scope().name) + 1:]
             for v in layer_variables])
        # including number of parameters
        out_str += ' ({})'.format(self.num_trainable_params())
        return out_str

    @property
    def rand_initializer(self):
        assert self._rand_initializer is not None
        return self._rand_initializer

    @property
    def rand_regularizer(self):
        return self._rand_regularizer

    @property
    def const_initializer(self):
        assert self._const_initializer is not None
        return self._const_initializer

    @property
    def const_regularizer(self):
        return self._const_regularizer

    @rand_initializer.setter
    def rand_initializer(self, value):
        self._rand_initializer = value

    @rand_regularizer.setter
    def rand_regularizer(self, value):
        self._rand_regularizer = value

    @const_initializer.setter
    def const_initializer(self, value):
        self._const_initializer = value

    @const_regularizer.setter
    def const_regularizer(self, value):
        self._const_regularizer = value
