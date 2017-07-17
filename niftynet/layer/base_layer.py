# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf
from niftynet.engine.restorer import RESTORABLE


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
    """
    Extends the Layer object to have trainable parameters,
    adding intiailizers and regularizers.
    """

    def __init__(self, name='trainable_op'):
        super(TrainableLayer, self).__init__(name=name)

        self._initializers = None
        self._regularizers = None

    def trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 self.layer_scope().name)

    def restore_from_checkpoint(self, checkpoint_name, scope=None):
        if scope is None:
            scope=self.layer_scope().name
        tf.add_to_collection(RESTORABLE,(self.layer_scope().name,
                                         checkpoint_name,scope))

    def regularizer_loss(self):
        return tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                                 self.layer_scope().name)

    def num_trainable_params(self):
        n = tf.Dimension(0)
        for x in self.trainable_variables():
            n += np.prod(x.get_shape())
        return int(n)

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
    def initializers(self):
        return self._initializers

    @property
    def regularizers(self):
        return self._regularizers

    @initializers.setter
    def initializers(self, value):
        assert isinstance(value, dict)
        self._initializers = value

    @regularizers.setter
    def regularizers(self, value):
        assert isinstance(value, dict)
        self._regularizers = value


class LayerFromCallable(Layer):
    """Module wrapping a function provided by the user.
    Analogous to snt.Module
    """

    def __init__(self, layer_op, name='untitled_op'):
        super(LayerFromCallable, self).__init__(name=name)
        if not callable(layer_op):
            raise TypeError("layer_op must be callable.")
        self._layer_op=layer_op

    def layer_op(self, *args, **kwargs):
        return self._layer_op(*args,**kwargs)
