from __future__ import absolute_import, division, print_function

import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.layer.layer_util import infer_spatial_rank
from niftynet.utilities.util_common import look_up_operations

COORDINATES_TYPE = tf.int32


class ResamplerLayer(Layer):
    """
    resample inputs according to sample_coords
    """

    def __init__(self,
                 interpolation="LINEAR",
                 boundary="ZERO",
                 name="resampler2d"):
        super(ResamplerLayer, self).__init__(name=name)
        self.boundary = boundary.upper()
        self.boundary_func = look_up_operations(
            self.boundary, SUPPORTED_BOUNDARY)
        self.interpolation = look_up_operations(
            interpolation.upper(), SUPPORTED_INTERPOLATION)

        if self.boundary == 'ZERO' and self.interpolation == 'BSPLINE':
            tf.logging.fatal('Zero padding is not supported for BSPLINE mode')
            raise NotImplementedError

    def layer_op(self, inputs, sample_coords):
        if self.interpolation == 'LINEAR':
            return self._resample_linear(inputs, sample_coords)
        if self.interpolation == 'NEAREST':
            return self._resample_nearest(inputs, sample_coords)
        if self.interpolation == 'BSPLINE':
            return self._resample_bspline(inputs, sample_coords)
        tf.logging.fatal('interplolation method not implmented')
        raise NotImplementedError

    def _resample_nearest(self, inputs, sample_coords):
        in_size = inputs.get_shape().as_list()
        batch_size = in_size[0]
        in_spatial_size = in_size[1:-1]

        out_size = sample_coords.get_shape().as_list()
        out_spatial_size = out_size[1:-1]
        out_spatial_rank = infer_spatial_rank(sample_coords)

        spatial_coords = self.boundary_func(
            tf.round(sample_coords), in_spatial_size)
        batch_ids = tf.reshape(
            tf.range(batch_size), [batch_size] + [1] * (out_spatial_rank + 1))
        batch_ids = tf.tile(batch_ids, [1] + out_spatial_size + [1])
        output = tf.gather_nd(
            inputs, tf.concat([batch_ids, spatial_coords], -1))

        if self.boundary == 'ZERO':
            scale = 1. / (tf.constant(in_spatial_size, dtype=tf.float32) - 1)
            mask = tf.logical_and(
                tf.reduce_all(sample_coords > 0,
                              axis=-1, keep_dims=True),
                tf.reduce_all(scale * sample_coords < 1,
                              axis=-1, keep_dims=True))
            return output * tf.to_float(mask)
        return output

    def _resample_linear(self, inputs, sample_coords):
        in_size = inputs.get_shape().as_list()
        in_spatial_size = in_size[1:-1]
        in_spatial_rank = infer_spatial_rank(inputs)
        batch_size = in_size[0]

        out_spatial_rank = infer_spatial_rank(sample_coords)
        out_spatial_size = sample_coords.get_shape().as_list()[1:-1]

        if in_spatial_rank == 2 and self.boundary == 'ZERO':
            inputs = tf.transpose(inputs, [0, 2, 1, 3])
            return tf.contrib.resampler.resampler(inputs, sample_coords)

        xy = tf.unstack(sample_coords, axis=-1)
        base_coords = [tf.floor(coords) for coords in xy]
        floor_coords = [self.boundary_func(x, in_spatial_size[idx])
                        for (idx, x) in enumerate(base_coords)]
        ceil_coords = [self.boundary_func(x + 1.0, in_spatial_size[idx])
                       for (idx, x) in enumerate(base_coords)]

        if self.boundary == 'ZERO':
            weight_0 = [tf.expand_dims(x - tf.cast(i, tf.float32), -1)
                        for (x, i) in zip(xy, floor_coords)]
            weight_1 = [tf.expand_dims(tf.cast(i, tf.float32) - x, -1)
                        for (x, i) in zip(xy, ceil_coords)]
        else:
            weight_0 = [tf.expand_dims(x - i, -1)
                        for (x, i) in zip(xy, base_coords)]
            weight_1 = [1.0 - w for w in weight_0]

        batch_ids = tf.reshape(
            tf.range(batch_size), [batch_size] + [1] * out_spatial_rank)
        batch_ids = tf.tile(batch_ids, [1] + out_spatial_size)
        sc = (floor_coords, ceil_coords)

        def get_knot(bc):
            coord = [sc[c][i] for i, c in enumerate(bc)]
            coord = tf.stack([batch_ids] + coord, -1)
            return tf.gather_nd(inputs, coord)

        def _pyramid_combination(samples, w_0, w_1):
            if len(w_0) == 1:
                return samples[0] * w_1[0] + samples[1] * w_0[0]
            f_0 = _pyramid_combination(samples[::2], w_0[:-1], w_1[:-1])
            f_1 = _pyramid_combination(samples[1::2], w_0[:-1], w_1[:-1])
            return f_0 * w_1[-1] + f_1 * w_0[-1]

        binary_neighbour_ids = [
            [int(c) for c in format(i, '0%ib' % in_spatial_rank)]
            for i in range(2 ** in_spatial_rank)]
        samples = [get_knot(bc) for bc in binary_neighbour_ids]
        return _pyramid_combination(samples, weight_0, weight_1)

    def _resample_bspline(self, inputs, sample_coords):
        in_size = inputs.get_shape().as_list()
        batch_size = in_size[0]
        in_spatial_size = in_size[1:-1]
        in_spatial_rank = infer_spatial_rank(inputs)

        out_spatial_rank = infer_spatial_rank(sample_coords)
        if in_spatial_rank == 2:
            raise NotImplementedError(
                'bspline interpolation not implemented for 2d yet')
        floor_coords = tf.floor(sample_coords)

        # Compute voxels to use for interpolation
        grid = tf.meshgrid([-1, 0, 1, 2],
                           [-1, 0, 1, 2],
                           [-1, 0, 1, 2],
                           indexing='ij')
        offset_shape = [1, -1] + [1] * out_spatial_rank + [in_spatial_rank]
        offsets = tf.reshape(tf.stack(grid, 3), offset_shape)
        spatial_coords = \
            offsets + tf.expand_dims(tf.cast(floor_coords, tf.int32), 1)
        spatial_coords = self.boundary_func(spatial_coords, in_spatial_size)
        knot_size = spatial_coords.get_shape().as_list()

        # Compute weights for each voxel
        def build_coef(u, d):
            coeff_list = [tf.pow(1 - u, 3),
                          3 * tf.pow(u, 3) - 6 * tf.pow(u, 2) + 4,
                          -3 * tf.pow(u, 3) + 3 * tf.pow(u, 2) + 3 * u + 1,
                          tf.pow(u, 3)]
            return tf.concat(coeff_list, d) / 6

        weight = tf.reshape(sample_coords - floor_coords, [batch_size, -1, 3])
        coef_shape = [batch_size, 1, 1, 1, -1]
        Bu = build_coef(tf.reshape(weight[:, :, 0], coef_shape), 1)
        Bv = build_coef(tf.reshape(weight[:, :, 1], coef_shape), 2)
        Bw = build_coef(tf.reshape(weight[:, :, 2], coef_shape), 3)
        all_weights = tf.reshape(Bu * Bv * Bw,
                                 [batch_size] + knot_size[1:-1] + [1])
        # Gather voxel values and compute weighted sum
        batch_coords = tf.reshape(
            tf.range(batch_size), [batch_size] + [1] * (len(knot_size) - 1))
        batch_coords = tf.tile(batch_coords, [1] + knot_size[1:-1] + [1])
        raw_samples = tf.gather_nd(
            inputs, tf.concat([batch_coords, spatial_coords], -1))
        return tf.reduce_sum(all_weights * raw_samples, reduction_indices=1)


def _boundary_replicate(sample_coords, input_size):
    sample_coords, input_size = _param_type_and_shape(sample_coords, input_size)
    return tf.maximum(tf.minimum(sample_coords, input_size - 1), 0)


def _boundary_circular(sample_coords, input_size):
    sample_coords, input_size = _param_type_and_shape(sample_coords, input_size)
    return tf.mod(tf.mod(sample_coords, input_size) + input_size, input_size)


def _boundary_symmetric(sample_coords, input_size):
    sample_coords, input_size = _param_type_and_shape(sample_coords, input_size)
    circular_size = input_size + input_size - 2
    return (input_size - 1) - tf.abs(
        (input_size - 1) - _boundary_circular(sample_coords, circular_size))


def _param_type_and_shape(sample_coords, input_size):
    sample_coords = tf.cast(sample_coords, COORDINATES_TYPE)
    try:
        input_size = tf.constant(input_size, dtype=COORDINATES_TYPE)
    except TypeError:
        pass
    # try: # broadcasting input_size to match the shape of coordinates
    #    if len(input_size) > 1:
    #        broadcasting_shape = [1] * (infer_spatial_rank(sample_coords) + 1)
    #        input_size = tf.reshape(input_size, broadcasting_shape + [-1])
    # except (TypeError, AssertionError):
    #    # do nothing
    #    pass
    return sample_coords, input_size


SUPPORTED_INTERPOLATION = {'BSPLINE', 'LINEAR', 'NEAREST'}

SUPPORTED_BOUNDARY = {
    'ZERO': _boundary_replicate,
    'REPLICATE': _boundary_replicate,
    'CIRCULAR': _boundary_circular,
    'SYMMETRIC': _boundary_symmetric}
