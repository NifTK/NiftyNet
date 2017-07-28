"""
Users can set window_size = 16, when the actual
spatial 3D window size is window_size = (16, 16, 16), this
simplifies the input in many cases and improves readability.

This utility expands the shortcut form params, to the fully specified
params accordingly
"""
import numpy as np


def expand_spatial_kernel_params(input_kernel_size, spatial_rank):
    """
    expand input parameter
    e.g., kernel_size = 3 is converted to kernel_size = (3, 3, 3)
    for 3D convolutions

    :param input_kernel_size:
    :param spatial_rank:
    :return: fully-specified kernel_size parameter
    """

    if (type(input_kernel_size) == int):
        return (input_kernel_size,) * spatial_rank
    else:
        input_kernel_size = np.asarray(input_kernel_size).flatten().tolist()
        assert len(input_kernel_size) == spatial_rank, \
            'param length should be the same as the spatial rank'
        return tuple(input_kernel_size)


def expand_padding_params(input_padding_value, spatial_rank):
    # TODO: to handle partially specified padding values
    # This function assumes 2.5D operates on the first two dims
    if spatial_rank == 3:
        return ((input_padding_value, input_padding_value),
                (input_padding_value, input_padding_value),
                (input_padding_value, input_padding_value))
    elif spatial_rank == 2:
        return ((input_padding_value, input_padding_value),
                (input_padding_value, input_padding_value))
    elif spatial_rank == 2.5:
        return ((input_padding_value, input_padding_value),
                (input_padding_value, input_padding_value),
                (1, 1))
