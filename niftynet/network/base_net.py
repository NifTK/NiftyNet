from __future__ import absolute_import, print_function

from niftynet.layer.base_layer import TrainableLayer


class BaseNet(TrainableLayer):
    """
    Template for networks
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name="net_template"):

        super(BaseNet, self).__init__(name=name)

        self.num_classes = num_classes
        self.acti_func = acti_func

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

        print('using {}'.format(name))