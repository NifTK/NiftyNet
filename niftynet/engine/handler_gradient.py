# -*- coding: utf-8 -*-
"""
This module implements a network model updater with gradient ops.
"""

import tensorflow as tf
import numpy as np

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

    def check_updates_param(self, sender, **msg):
        """
        This function checks if the providing rules for training updates
        are consistent and valid and throws errors or warning if the
        situation warrants it
        :param sender:
        :return:
        """
        if sender.training_values is None:
            sender.training_values = []
        if sender.training_types is None:
            sender.training_types = []
        gradient_ops = sender.gradient_op
        training_types = sender.training_types
        training_values = np.asarray(sender.training_values)
        if sender.training_item is None:
            sender.training_item = 0
        if sender.training_mode is None:
            sender.training_mode = 0
        if len(training_values) != len(training_types):
            tf.logging.fatal("The number of values for training updates (%d) "
                             "is "
                             "different that the number of update types (%d) "
                             % (len(training_values), len(training_types)))
            raise ValueError
        if sender.training_list is None or len(sender.training_list)==0:
            sender.training_list = np.arange(sender.training_mode,
                                             len(gradient_ops))
        if sender.training_list is not None and len(sender.training_list)>0 \
                and sender.training_mode is not None:
            sender.training_list = [sender.training_mode,
                                    ]+ list(sender.training_list)
            print(sender.training_list)
        if sender.training_item is not None:
            if sender.training_item >= len(sender.training_list):
                tf.logging.fatal(
                    "The number of values for training updates (%d) "
                    "is not appropriate given the training item (%d) "
                    % (len(sender.training_list), sender.training_item))
                raise ValueError
            if sender.training_list[sender.training_item] != \
                    sender.training_mode:
                tf.logging.warning("Updating training mode from %d to %d "
                                   % (sender.training_mode,
                                      sender.training_list[
                                          sender.training_item]))
                sender.training_mode = sender.training_list[sender.training_item]
        if len(training_values) < len(gradient_ops) - 1:
            tf.logging.warning("The last updates of training may not be "
                               "reached since the number of updates is %d "
                               "and the number of training modes is %d "
                               % (len(training_values), len(gradient_ops)))
            if len(training_values) == 1:
                sender.training_values = np.asarray(list(training_values) *
                                                     len(
                    sender.training_list))

                sender.training_types = training_types * len(
                    sender.training_list)
                tf.logging.warning("Same update criterion chosen for each "
                                   "stage: %s %f" %(sender.training_types[0],
                                                    sender.training_values[0]))

        for (type_train, value_train, i_update) in zip(training_types,
                                                       training_values,
                                                       np.arange(len(
                                                           training_values))):
            if type_train == 'time' and value_train < 2:
                    tf.logging.fatal("Incompatibility between training_values"
                                     " and training type with value %f and "
                                     "type %s at training update %d"
                                     % (value_train, type_train, i_update))
                    raise ValueError

        return

    def make_gradients_op(self, sender, **_unused):
        """
        Making ``optimiser.apply_gradients`` ops. Create an array of gradient
         ops that can be called according to the training mode

        :param sender: a niftynet.application instance
        :param _unused:
        :return:
        """
        with tf.name_scope('ApplyGradients'):
            gradients_array = sender.gradients_collector.gradients
            depth_gradients_array = util_common.list_depth_count(
                gradients_array)
            if depth_gradients_array == 2:
                true_gradients_array = [gradients_array]
            elif depth_gradients_array == 4:
                true_gradients_array = gradients_array[0]
            else:
                true_gradients_array = gradients_array

            sender.gradient_op = []

            bn_ops = tf.get_collection(BN_COLLECTION, PRIMARY_NAME_SCOPE)
            if not bn_ops:
                for ops_item in range(0, len(true_gradients_array)):
                    sender.gradient_op.append(_apply_gradients(
                        sender.optimiser, true_gradients_array[ops_item]))

            else:
                with tf.get_default_graph().control_dependencies(bn_ops):
                    for ops_item in range(0, len(true_gradients_array)):
                        sender.gradient_op.append(_apply_gradients(
                            sender.optimiser, true_gradients_array[ops_item]))
            self.check_updates_param(sender)

    def add_gradients(self, sender, **msg):
        """
        Event handler to add gradients to iteration message ops_to_run.
        Update the training mode according to the rules specified for the
        application

        See also
        ``niftynet.application.base_application.set_network_gradient_op``

        :param sender: a niftynet.application instance
        :param msg: an iteration message instance
        :return:
        """

        if sender.training_mode is None:
            sender.training_mode = 0
        if sender.action_param.starting_iter + 1 == msg[
            'iter_msg'].current_iter:
            sender.training_values = np.asarray(sender.training_values)
            if len(sender.training_types) > 0 and sender.training_types[
                       sender.training_item] == 'time':
                sender.training_values[sender.training_item] = \
                    sender.training_values[sender.training_item] + msg[
                        'iter_msg'].current_iter
                print("new max iter updated at start is %d" %
                      sender.training_values[sender.training_item])
        if len(sender.gradient_op) > 0 and sender.training_mode < len(
                sender.gradient_op)-1:
            print("need to check if update is possible",
                  sender.training_mode, sender.training_types[
                      sender.training_item])
            if sender.performance_history is not None:
                print('len perf ', len(sender.performance_history))
            else:
                print("Performance history is empty")

            if 'time' not in sender.training_types[sender.training_item]:
                self.update_training_mode_perfbased(sender,
                                                    thresh=sender.
                                                    training_values
                                                    [sender.training_item],
                                                    patience=2,
                                                    mode=sender.training_types[
                                                        sender.training_item],
                                                    **msg)
            else:
                self.update_training_mode_timebased(sender, time=
                                                    sender.training_values[
                                                      sender.training_item]
                                                    , **msg)

        if msg['iter_msg'].is_training:
            print("training mode is %d" % sender.training_mode)
            msg['iter_msg'].ops_to_run['gradients'] = sender.gradient_op[
                sender.training_mode]
            # sender.outputs_collector.inside_vars['train_mode'] = \
            #     tf.constant(sender.training_mode)
            # sender.outputs_collector.console_vars['train_mode_test'] = \
            #     tf.constant(sender.training_mode)


    def update_training_mode_timebased(self, sender, time, **msg):
        """
        Update the training mode when the update is based on the number of
        iterations
        :param sender: application
        :param time: time (iteration limit) at which an update in the
        training should be made
        :param msg: Contains information relative to the current state
        :return:
        """
        if msg['iter_msg'].current_iter <= time:
            # print(msg['iter_msg'].current_iter, 'but limit is ', time)
            return
        else:
            # print("Limit for training time updated reached")
            tf.logging.warning("Updating of training mode after %d "
                               "operations" % time)
            sender.training_item += 1
            if sender.training_item < len(sender.training_list):
                sender.training_mode = sender.training_list[sender.training_item]
                # sender.outputs_collector.inside_vars['train_mode'] = \
                #     sender.training_mode
            else:
                tf.logging.warning("Maximum update reached from training "
                                   "list, no further mode update")
                sender.training_item -= 1
                msg['iter_msg'].should_stop = True
            # Update of all training value times to get relative wait

            if sender.training_types[sender.training_item] == 'time':
                sender.training_values[sender.training_item] += msg[
                    'iter_msg'].current_iter
                tf.logging.warning("updating max iter for update to %d" %
                sender.training_values[sender.training_item])
                # sender.outputs_collector.console_vars['train_mode_test'] = \
                #     sender.training_mode
                # sender.outputs_collector.inside_vars['train_mode'] = \
                #     sender.training_mode

            return

    def update_training_mode_perfbased(self, sender, thresh, patience, mode,
                                       **msg):
        """
        Update the training mode according to the performance history. Clean
        the performance buffer after each update
        :param sender: application
        :param thresh: threshold on which the update is made
        :param patience: number of iterations to look at in the history to
        take a decision
        :param mode: way of looking at the performance over the last (
        patience) number of iterations
        :return:
        """
        if sender.performance_history is None:
            sender.performance_history = []
        if len(sender.performance_history) < patience:
            return
        performance_to_consider = sender.performance_history[-patience:]

        if mode == 'max':
            value = tf.reduce_max(performance_to_consider)
        elif mode == 'perc':
            value = np.abs((np.max(performance_to_consider) - np.min(
                performance_to_consider))/np.max(
                    performance_to_consider))
        elif mode == 'robust_perc':
            perc = np.percentile(performance_to_consider, q=[5, 95])
            value = np.abs((perc[0] - perc[1]) / perc[1])
        elif mode == 'mean':
            value = np.mean(performance_to_consider)
        else:
            value = np.mean(performance_to_consider)
        print('value is %f and target is %f' % (value, thresh))
        if value < thresh:
            sender.training_item += 1
            if sender.training_item < len(sender.training_list):
                sender.training_mode = sender.training_list[sender.training_item]
                # sender.outputs_collector.inside_vars['train_mode'] = \
                #     tf.constant(sender.training_mode)
            else:
                tf.logging.warning("Maximum update reached from training "
                                   "list, no further mode update")
                sender.training_item -= 1
                msg['iter_msg'].should_stop = True
            tf.logging.warning("Going on to next training phase %d since "
                               "value is %f and target is %f"
                               % (sender.training_mode, value, thresh))
            sender.performance_history = []
            # sender.outputs_collector.inside_vars['train_mode'] = tf.constant(
            #     sender.training_mode)
            if sender.training_types[sender.training_item] == 'time':
                sender.training_values[sender.training_item] += msg[
                    'iter_msg'].current_iter


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
