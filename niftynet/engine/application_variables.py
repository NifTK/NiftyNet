# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import PIL
import numpy as np
import tensorflow as tf
from PIL.GifImagePlugin import Image as GIF
from tensorflow.contrib.framework import list_variables
from tensorflow.core.framework import summary_pb2

from niftynet.utilities import util_common as util
from niftynet.utilities.restore_initializer import restore_initializer
from niftynet.utilities.util_common import look_up_operations


def image3_sagittal(name, tensor, max_outputs=3,
                    collections=[tf.GraphKeys.SUMMARIES]):
    return image3(name, tensor, max_outputs, collections, [1], [2, 3])


def image3_coronal(name, tensor, max_outputs=3,
                   collections=[tf.GraphKeys.SUMMARIES]):
    return image3(name, tensor, max_outputs, collections, [2], [1, 3])


def image3_axial(name, tensor, max_outputs=3,
                 collections=[tf.GraphKeys.SUMMARIES]):
    return image3(name, tensor, max_outputs, collections, [3], [1, 2])


RESTORABLE = 'NiftyNetObjectsToRestore'

CONSOLE = 'niftynetconsole'
TF_SUMMARIES = tf.GraphKeys.SUMMARIES
SUPPORTED_SUMMARY = {'scalar': tf.summary.scalar,
                     'histogram': tf.summary.histogram,
                     'image3_sagittal': image3_sagittal,
                     'image3_coronal': image3_coronal,
                     'image3_axial': image3_axial}


class GradientsCollector(object):
    def __init__(self, n_devices=1):
        self._gradients = []
        self.n_devices = n_devices

    def add_to_collection(self, gradients):
        self._gradients.append(gradients)
        assert self.current_tower_id <= self.n_devices, \
            "call add_to_collection once per device"

    @property
    def current_tower_id(self):
        return len(self._gradients)

    @property
    def gradients(self):
        # return averaged over devices
        assert self._gradients, \
            "Please add gradients to collector when constructing the graph"
        return util.average_gradients(self._gradients)


class OutputsCollector(object):
    def __init__(self, n_devices=1):
        self.console_vars = {}
        self.tf_summary_vars = {}
        self._merge_op = None

        self.n_devices = n_devices

    def _add_to_dict(self, var_dict, var, name, do_averaging):
        """
        update the dict, return the variable name if the variable is
        ready to run (by tf.Session).
        """
        assert isinstance(var, tf.Tensor), \
            "only supports adding one tf.Tensor at a time," \
            "but received {}".format(var)

        if do_averaging and self.n_devices > 1:
            # collecting variables across devices as a list
            var_list = var_dict.get(name, [])
            assert isinstance(var_list, list), \
                "averaged variable name {} has been taken".format(name)
            var_list.append(var)
            var_dict[name] = var_list
            assert len(var_list) <= self.n_devices, \
                "averaged variable {} has been used " \
                "in the collector".format(name)
        else:
            # collecting variables and rename if exists
            _uniq_id = 0
            new_name = name
            while new_name in var_dict:
                _uniq_id += 1
                new_name = '{}_{}'.format(name, _uniq_id)
            var_dict[new_name] = var

    def add_to_collection(self, var, name,
                          average_over_devices=False,
                          collection=CONSOLE, **kwargs):
        if collection == CONSOLE:
            self._add_to_console(var, name, average_over_devices)
        if collection == TF_SUMMARIES:
            self._add_to_tf_summary(var, name, average_over_devices, **kwargs)

    def _add_to_console(self, var, name, average_over_devices=False):
        self._add_to_dict(self.console_vars,
                          var, name, average_over_devices)

    def _add_to_tf_summary(self, var, name,
                           average_over_devices=False,
                           summary_type='scalar'):
        summary_op = look_up_operations(summary_type, SUPPORTED_SUMMARY)
        self._add_to_dict(self.tf_summary_vars,
                          var, name, average_over_devices)
        values = self.tf_summary_vars.get(name, None)
        if isinstance(values, tf.Tensor):
            summary_op(name=name, tensor=values, collections=[TF_SUMMARIES])

    def variables(self, collection=CONSOLE):
        if collection == CONSOLE:
            return self.console_vars
        elif collection == TF_SUMMARIES:
            return self._merge_op if self._merge_op is not None else {}
        else:
            raise ValueError("unknown output variables type_str")

    def finalise_output_op(self):
        """
        This function checks the dictionary, if the variable needs to
        be averaged over devices, then a reduce_mean node is added to
        the graph.
        This function should be called in
        ApplicationDriver.create_graph function
        """
        for var_name in self.console_vars:
            values = self.console_vars.get(var_name, None)
            if isinstance(values, list):
                self.console_vars[var_name] = tf.reduce_mean(
                    values, name=var_name)

        for var_name in self.tf_summary_vars:
            values = self.tf_summary_vars.get(var_name, None)
            if isinstance(values, list):
                self.tf_summary_vars[var_name] = tf.reduce_mean(
                    values, name=var_name)
                summary_name = '{}_device_average_'.format(var_name)
                tf.summary.scalar(name=summary_name,
                                  tensor=self.tf_summary_vars[var_name],
                                  collections=[TF_SUMMARIES])
        self._merge_op = tf.summary.merge_all(key=TF_SUMMARIES)


def image3_animatedGIF(tag, ims):
    # x=numpy.random.randint(0,256,[10,10,10],numpy.uint8)
    ims = [np.asarray((ims[i, :, :]).astype(np.uint8))
           for i in range(ims.shape[0])]
    ims = [GIF.fromarray(im) for im in ims]
    s = b''
    for b in PIL.GifImagePlugin.getheader(ims[0])[0]:
        s += b
    s += b'\x21\xFF\x0B\x4E\x45\x54\x53\x43\x41\x50\x45\x32\x2E\x30\x03\x01\x00\x00\x00'
    for i in ims:
        for b in PIL.GifImagePlugin.getdata(i):
            s += b
    s += b'\x3B'
    image_summary = summary_pb2.Summary.Value(
        tag=tag,
        image=summary_pb2.Summary.Image(height=10,
                                        width=10,
                                        colorspace=1,
                                        encoded_image_string=str(s)))
    return [summary_pb2.Summary(value=[image_summary]).SerializeToString()]


def image3(name,
           tensor,
           max_outputs=3,
           collections=[tf.GraphKeys.SUMMARIES],
           animation_axes=[1],
           image_axes=[2, 3],
           other_indices={}):
    ''' Summary for higher dimensional images
    Parameters:
    name: string name for the summary
    tensor:   tensor to summarize. Should be in the range 0..255.
              By default, assumes tensor is NDHWC, and animates (through D)
              HxW slices of the 1st channel.
    collections: list of strings collections to add the summary to
    animation_axes=[1],image_axes=[2,3]
    '''
    if max_outputs == 1:
        suffix = '/image'
    else:
        suffix = '/image/{}'
    axis_order = [0] + animation_axes + image_axes
    # slice tensor
    slicing = tuple((slice(None) if i in axis_order else slice(
        other_indices.get(i, 0), other_indices.get(i, 0) + 1) for i in
                     range(len(tensor.shape))))
    tensor = tensor[slicing]
    axis_order_all = axis_order + [i for i in range(len(tensor.shape.as_list()))
                                   if i not in axis_order]
    new_shape = [tensor.shape.as_list()[0], -1,
                 tensor.shape.as_list()[axis_order[-2]],
                 tensor.shape.as_list()[axis_order[-1]]]
    transposed_tensor = tf.reshape(tf.transpose(tensor, axis_order_all),
                                   new_shape)
    # split images
    with tf.device('/cpu:0'):
        for it in range(min(max_outputs, transposed_tensor.shape.as_list()[0])):
            T = tf.py_func(image3_animatedGIF,
                           [name + suffix.format(it),
                            transposed_tensor[it, :, :, :]],
                           tf.string)
            [tf.add_to_collection(c, T) for c in collections]
    return T


def resolve_checkpoint(checkpoint_name):
    # For now only supports checkpoint_name where
    # checkpoint_name.index is in the file system
    # eventually will support checkpoint names that can be referenced
    # in a paths file
    if os.path.exists(checkpoint_name + '.index'):
        return checkpoint_name
    raise ValueError('Invalid checkpoint {}'.format(checkpoint_name))


def global_variables_initialize_or_restorer(var_list=None):
    # For any scope added to RESTORABLE collection:
    # variable will be restored from a checkpoint if it exists in the
    # specified checkpoint and no scope ancestor can restore it.
    if var_list is None:
        var_list = tf.global_variables()
    restorable = sorted(tf.get_collection(RESTORABLE), key=lambda x: x[0])
    restored_vars = {}
    for scope, checkpoint_name, checkpoint_scope in restorable:
        variables_in_scope = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
        checkpoint_file = resolve_checkpoint(checkpoint_name)
        variables_in_file = [v for v, s in list_variables(checkpoint_file)]
        rename = lambda x: x.replace(scope, checkpoint_scope).replace(':0', '')
        to_restore = [v for v in variables_in_scope
                      if v in var_list and rename(v.name) in variables_in_file]
        for var in to_restore:
            if var in restored_vars:
                continue
            if '/' in rename(var.name):
                checkpoint_subscope, var_name = rename(var.name).rsplit('/', 1)
            else:
                checkpoint_subscope, var_name = None, rename(var.name)
            initializer = restore_initializer(
                checkpoint_name, var_name, checkpoint_subscope)
            restored_vars[var] = tf.assign(var,
                                           initializer(var.get_shape(),
                                                       dtype=var.dtype))
    init_others = tf.variables_initializer(
        [v for v in var_list if v not in restored_vars])
    restore_op = tf.group(init_others, *list(restored_vars.values()))
    return restore_op
