# -*- coding: utf-8 -*-
"""
This module implements a network model updater with gradient ops.
"""

import tensorflow as tf

from niftynet.engine.signal import ITER_STARTED, GRAPH_CREATED
from niftynet.layer.bn import BN_COLLECTION
from niftynet.utilities import util_common

PRIMARY_NAME_SCOPE = 'worker_0'


class ApplyGradients(object):
    """
    This class handles iteration events to update the model with gradient op
    (by setting iteration message with a 'gradients' op at the beginning of
    each iteration).
    """

    def __init__(self, is_training_action=False, **_unused):
        if not is_training_action:
            return
        GRAPH_CREATED.connect(self.make_gradients_op)
        ITER_STARTED.connect(self.add_gradients)

    def make_gradients_op(self, sender, **_unused):
        """
        Making ``optimiser.apply_gradients`` ops.

        :param sender: a niftynet.application instance
        :param _unused:
        :return:
        """
        with tf.name_scope('ApplyGradients'):
            gradients = sender.gradients_collector.gradients
            bn_ops = tf.get_collection(BN_COLLECTION, PRIMARY_NAME_SCOPE)
            if not bn_ops:
                sender.gradient_op = _apply_gradients(
                    sender.optimiser, gradients)
            else:
                with tf.get_default_graph().control_dependencies(bn_ops):
                    sender.gradient_op = _apply_gradients(
                        sender.optimiser, gradients)

    def add_gradients(self, sender, **msg):
        """
        Event handler to add gradients to iteration message ops_to_run.

        See also
        ``niftynet.application.base_application.set_network_gradient_op``

        :param sender: a niftynet.application instance
        :param msg: an iteration message instance
        :return:
        """
        if msg['iter_msg'].is_training:
            msg['iter_msg'].ops_to_run['gradients'] = sender.gradient_op


def _apply_gradients(optimiser, gradients):
    """
    Create gradient op by ``optimiser.apply_gradients``.
    This function sets ``self.gradient_op``.

    Override this function for more complex optimisations such as
    using different optimisers for sub-networks.

    :param gradients: processed gradients from the gradient_collector
    :return:
    """
    grad_list_depth = util_common.list_depth_count(gradients)
    if grad_list_depth == 3:
        # nested depth 3 means: gradients list is nested in terms of:
        # list of networks -> list of network variables
        return [optimiser.apply_gradients(grad) for grad in gradients]
    elif grad_list_depth == 2:
        # nested depth 2 means:
        # gradients list is a list of variables
        return optimiser.apply_gradients(gradients)
    raise NotImplementedError(
        'This app supports updating a network, or a list of networks.')
