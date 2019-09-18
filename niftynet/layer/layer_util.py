# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np


def check_spatial_dims(input_tensor, criteria):
    """
    valid each of the spatial dims against `criteria`
    criteria can be a lambda function
    e.g. lambda x : x > 10 checks whether each dim is greater than 10
    """
    input_shape = input_tensor.shape
    if not input_shape.is_fully_defined():
        # skip checking if the input has dynamic shapes
        return True
    input_shape.with_rank_at_least(3)
    spatial_dims = input_shape[1:-1].as_list()
    all_dims_satisfied = np.all([criteria(x) for x in spatial_dims])
    if not all_dims_satisfied:
        import inspect
        raise ValueError("input tensor's spatial dimensionality"
                         " not compatible, please tune "
                         "the input window sizes. "
                         "(e.g. lambda x : x % 8 == 0 checks whether each dimension is divisible by 8)\n{}".format(
            inspect.getsource(criteria)))
    return all_dims_satisfied


def infer_spatial_rank(input_tensor):
    """
    e.g. given an input tensor [Batch, X, Y, Z, Feature] the spatial rank is 3
    """
    input_shape = input_tensor.shape
    input_shape.with_rank_at_least(3)
    #dims = input_tensor.get_shape().ndims - 2
    #assert dims > 0, "input tensor should have at least one spatial dim, " \
    #                 "in addition to batch and channel dims"
    return int(input_shape.ndims - 2)


def trivial_kernel(kernel_shape):
    """
    This function generates a trivial kernel with all 0s except for the
    element in its spatial center
    e.g. trivial_kernel((3, 3, 1, 1,)) returns a kernel of::

        [[[[0]], [[0]], [[0]]],
         [[[0]], [[1]], [[0]]],
         [[[0]], [[0]], [[0]]]]

    kernel_shape[-1] and kernel_shape[-2] should be 1, so that it operates
    on the spatial dims only.  However, there is no exact spatial centre
    if np.any((kernel_shape % 2) == 0). This is fine in many cases
    as np.sum(trivial_kernel(kernel_shape)) == 1
    """
    assert kernel_shape[-1] == 1
    assert kernel_shape[-2] == 1
    # assert np.all((kernel_shape % 2) == 1)
    kernel = np.zeros(kernel_shape)
    flattened = kernel.reshape(-1)
    flattened[np.prod(kernel_shape) // 2] = 1
    return flattened.reshape(kernel_shape)


def expand_spatial_params(input_param, spatial_rank, param_type=int):
    """
    Expand input parameter
    e.g., ``kernel_size=3`` is converted to ``kernel_size=[3, 3, 3]``
    for 3D images (when ``spatial_rank == 3``).
    """
    spatial_rank = int(spatial_rank)
    try:
        if param_type == int:
            input_param = int(input_param)
        else:
            input_param = float(input_param)
        return (input_param,) * spatial_rank
    except (ValueError, TypeError):
        pass
    try:
        if param_type == int:
            input_param = \
                np.asarray(input_param).flatten().astype(np.int).tolist()
        else:
            input_param = \
                np.asarray(input_param).flatten().astype(np.float).tolist()
    except (ValueError, TypeError):
        # skip type casting if it's a TF tensor
        pass
    assert len(input_param) >= spatial_rank, \
        'param length should be at least have the length of spatial rank'
    return tuple(input_param[:spatial_rank])

# class RequireKeywords(object):
#    def __init__(self, *list_of_keys):
#        self.keys = list_of_keys
#
#    def __call__(self, f):
#        def wrapped(*args, **kwargs):
#            for key in self.keys:
#                if key not in kwargs:
#                    raise ValueError("{}: specify keywords: '{}'".format(
#                        args[0].layer_scope().name, self.keys))
#            return f(*args, **kwargs)
#        return wrapped


def check_divisible_channels(input_tensor, n_channel_splits):
    """
    Check if the number of channels (last dim) of the input tensor
    is divisible by ``n_channel_splits``. If True, returns
    ``n_input_channels / n_channel_splits``, raises AssertionError otherwise

    :param input_tensor:
    :param n_channel_splits:
    :return: n_input_channels / n_channel_splits
    """

    n_input_channels = int(input_tensor.shape.as_list()[-1])
    n_channel_splits = int(n_channel_splits)
    assert n_channel_splits > 0 and n_input_channels % n_channel_splits == 0, \
        "Number of feature channels should be divisible by " \
        "n_channel_splits {}, so that given an input with n_input_channels, " \
        "the output tensor will have " \
        "n_input_channels / n_channel_splits.".format(n_channel_splits)
    return n_input_channels / n_channel_splits
