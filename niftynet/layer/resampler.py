# -*- coding: utf-8 -*-
"""
Resampler layer initially implemented in
https://github.com/niftk/NiftyNet/blob/v0.2.0.post1/niftynet/layer/spatial_transformer.py
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

from niftynet.layer.base_layer import Layer
from niftynet.layer.layer_util import infer_spatial_rank
from niftynet.utilities.util_common import look_up_operations

COORDINATES_TYPE = tf.int32
EPS = 1e-6


class ResamplerLayer(Layer):
    """
    resample inputs according to ``sample_coords``
    """

    def __init__(self,
                 interpolation="LINEAR",
                 boundary="REPLICATE",
                 name="resampler",
                 implementation="Fast"):
        super(ResamplerLayer, self).__init__(name=name)
        self.boundary = boundary.upper()
        self.boundary_func = look_up_operations(
            self.boundary, SUPPORTED_BOUNDARY)
        self.interpolation = look_up_operations(
            interpolation.upper(), SUPPORTED_INTERPOLATION)

        if self.boundary == 'ZERO' and self.interpolation == 'BSPLINE':
            tf.logging.fatal('Zero padding is not supported for BSPLINE mode')
            raise NotImplementedError

        if self.boundary == 'ZERO' and self.interpolation == 'IDW':
            tf.logging.warning('Zero padding is not supported for IDW mode')
            # raise NotImplementedError

        self.FastResamplerLayer = None  #
        if implementation.lower() in ['niftyreg', 'fast']:
            # check if niftyreg_resampling_layer is installed
            try:
                from niftyreg_image_resampling import NiftyregImageResamplingLayer
                import niftyreg_image_resampling as resampler_module
            except ImportError:
                tf.logging.warning('''
                    niftyreg_image_resampling is not installed; falling back onto
                    niftynet.layer.resampler.ResamplerLayer. To allow fast resampling,
                    please see installation instructions in
                    niftynet/contrib/niftyreg_image_resampling/README.md
                    ''')
                return

            # Passthrough of supported boundary types for  resampling
            SUPPORTED_BOUNDARY_FAST = resampler_module.SUPPORTED_BOUNDARY

            # Passthrough of supported interpolation types for NiftyReg resampling
            SUPPORTED_INTERPOLATION_FAST = resampler_module.SUPPORTED_INTERPOLATION
            # check compatibility of the resampling options with niftyreg_image_resampling
            try:
                boundary_fast = look_up_operations(self.boundary, SUPPORTED_BOUNDARY_FAST)
                interp_fast = look_up_operations(self.interpolation, SUPPORTED_INTERPOLATION_FAST)
                self.FastResamplerLayer = NiftyregImageResamplingLayer(interp_fast, boundary_fast)
                tf.logging.info('''NiftyReg image resampling is used.''')
            except ValueError as e:
                tf.logging.warning(e)
                tf.logging.warning('''Falling back onto niftynet.layer.resampler.ResamplerLayer.''')

    def layer_op(self, inputs, sample_coords):
        """
        This layer resamples 2D or 3D data given the coordinates.

        In terms of 3D inputs,

        when the shape of ``inputs`` is ``[batch, x, y, z, num_channels]``,
        the shape of ``sample_coords`` can be
        ``[1, d0, d1, ..., 3]`` or ``[batch, d0, d1, ..., 3]``.

        The output shape would be ``[batch, d0, d1, ..., num_channels]``.

        Similarly, in 2D,

        when the shape of ``inputs`` is ``[batch, x, y, num_channels]``,
        the shape of ``sample_coords`` can be
        ``[1, d0, d1, ..., 2]`` or ``[batch, d0, d1, ..., 2]``.

        The output shape would be ``[batch, d0, d1, ... num_channels]``

        (If the shape of ``inputs`` is not fully specified, ``sample_coords``
        must be checked before using this function, to make sure the
        coordinates are pointing to locations within the inputs.)

        (Resampling 2D inputs is implemented by calling
        ``tf.contrib.resampler.resampler``. The interpretaion of coordinates is
        different in between this function and
        ``tf.contrib.resampler.resampler``:
        using ``self.layer_op(inputs, sample_coords)`` for 2D data
        is equivalent to (apart from the batch size broadcasting feature)::

            tf.contrib.resampler.resampler(
                tf.transpose(inputs, [0, 2, 1, 3]), sample_coords)

        (No gradient is computed for ``NEAREST`` method, and
         some of the padding modes.)
        """

        # check the input dims
        try:
            batch_inputs = int(inputs.shape[0])
            batch_sample_coords = int(sample_coords.shape[0])
        except (TypeError, ValueError):
            tf.logging.fatal('Unknown input shape, at least batch size '
                             'needs to be specified.')
            raise

        if batch_inputs != batch_sample_coords and batch_sample_coords > 1:
            tf.logging.fatal(
                '\nOnly the following two cases are currently supported:\n'
                '    - batch size of inputs == batch size of sample_coords\n'
                '    - batch size of sample_coords == 1\n'
                'In the second case, sample_coords will be applied to each of '
                'the batch component of the inputs.')
            raise ValueError

        # input_spatial_rank = infer_spatial_rank(inputs)
        # if input_spatial_rank != 2 and input_spatial_rank != 3:
        #     tf.logging.fatal('Only 2D or 3D inputs are supported.')
        #     raise ValueError

        try:
            coords_n_dim = int(sample_coords.shape[-1])
        except (TypeError, ValueError):
            tf.logging.fatal(
                'The last dim of the coordinates must have 2 or 3 elements.')
            raise

        if infer_spatial_rank(inputs) != coords_n_dim:
            tf.logging.fatal(
                'sample_coords.shape[-1] must be the same as the spatial rank '
                'of the inputs.')
            raise ValueError

        # currently converting everything to floats
        if inputs.dtype not in SUPPORTED_INPUT_DTYPE:
            inputs = tf.to_float(inputs)
        if sample_coords.dtype not in SUPPORTED_INPUT_DTYPE:
            sample_coords = tf.to_float(sample_coords)

        # use fast resampling layer if available
        if self.FastResamplerLayer is not None:
            return self.FastResamplerLayer.layer_op(inputs, sample_coords)
        # otherwise compatibility resampling layer is used
        if self.interpolation == 'LINEAR':
            return self._resample_linear(inputs, sample_coords)
        if self.interpolation == 'NEAREST':
            return self._resample_nearest(inputs, sample_coords)
        if self.interpolation == 'BSPLINE':
            return self._resample_bspline(inputs, sample_coords)
        if self.interpolation == 'IDW':
            return self._resample_inv_dst_weighting(inputs, sample_coords)
        tf.logging.fatal('interpolation method not implmented')
        raise NotImplementedError

    def _resample_nearest(self, inputs, sample_coords):
        # This is forward only as no gradient for tf.round

        # read input shape
        in_size = inputs.shape
        partial_shape = not in_size.is_fully_defined()
        in_spatial_size = None

        try:
            batch_size = int(in_size[0])
            n_coords = int(sample_coords.shape[0])
        except (TypeError, AssertionError, ValueError):
            tf.logging.fatal('Unknown input shape, at least batch size '
                             'and rank of the inputs are required.')
            raise

        # quantise coordinates
        spatial_coords = tf.round(sample_coords)
        if not partial_shape:
            in_spatial_size = in_size.as_list()[1:-1]
            spatial_coords = self.boundary_func(
                spatial_coords, in_spatial_size)
        spatial_coords = tf.cast(spatial_coords, COORDINATES_TYPE)

        if batch_size == n_coords:
            batch_inputs = tf.unstack(inputs)
            batch_coords = tf.unstack(spatial_coords)
            gathered_image = [tf.gather_nd(img, coord) for (img, coord) in
                              zip(batch_inputs, batch_coords)]
        elif n_coords == 1 and batch_size > 1:
            gathered_image = [tf.gather_nd(img, spatial_coords[0])
                              for img in tf.unstack(inputs)]
        else:
            raise NotImplementedError
        output = tf.stack(gathered_image, axis=0)

        if self.boundary == 'ZERO' and in_spatial_size:
            scale = 1. / (tf.constant(in_spatial_size, dtype=tf.float32) - 1.)
            mask = tf.logical_and(
                tf.reduce_all(sample_coords > 0, -1, True),
                tf.reduce_all(scale * sample_coords < 1, -1, True))
            return output * tf.to_float(mask)
        return output

    def _resample_linear(self, inputs, sample_coords):
        """
        Bilinear or trilinear resampling.

        :param inputs:
        :param sample_coords:
        :return:
        """

        # read input shape
        in_size = inputs.shape
        partial_shape = not in_size.is_fully_defined()

        try:
            batch_size = int(in_size[0])
            n_coords = int(sample_coords.shape[0])
            in_spatial_rank = infer_spatial_rank(inputs)
            in_spatial_size = \
                None if partial_shape else in_size.as_list()[1:-1]
        except (TypeError, AssertionError, ValueError):
            tf.logging.fatal('Unknown input shape, at least batch size '
                             'and rank of the inputs are required.')
            raise

        # read output shape
        out_spatial_rank = infer_spatial_rank(sample_coords)
        out_spatial_size = sample_coords.shape.as_list()[1:-1]

        if in_spatial_rank == 2 and self.boundary == 'ZERO':
            # calling TF's resampler
            inputs = tf.transpose(inputs, [0, 2, 1, 3])
            if batch_size == n_coords:
                return tf.contrib.resampler.resampler(inputs, sample_coords)
            outputs = [
                tf.contrib.resampler.resampler(
                    tf.expand_dims(img), sample_coords)
                for img in tf.unstack(inputs)]
            return tf.concat(outputs, axis=0)

        xy = tf.unstack(sample_coords, axis=-1)
        base_coords = [tf.floor(coords) for coords in xy]
        if partial_shape:
            # if input shape is not defined, unable to compute
            # boundary elements
            floor_coords = [coord for coord in base_coords]
            ceil_coords = [coord + 1.0 for coord in base_coords]
        else:
            floor_coords = [self.boundary_func(x, in_spatial_size[idx])
                            for (idx, x) in enumerate(base_coords)]
            ceil_coords = [self.boundary_func(x + 1.0, in_spatial_size[idx])
                           for (idx, x) in enumerate(base_coords)]

        if self.boundary == 'ZERO':
            weight_0 = [tf.expand_dims(x - i, -1)
                        for (x, i) in zip(xy, floor_coords)]
            weight_1 = [tf.expand_dims(i - x, -1)
                        for (x, i) in zip(xy, ceil_coords)]
        else:
            weight_0 = [tf.expand_dims(x - i, -1)
                        for (x, i) in zip(xy, base_coords)]
            weight_1 = [1.0 - w for w in weight_0]

        sc = (tf.cast(floor_coords, COORDINATES_TYPE),
              tf.cast(ceil_coords, COORDINATES_TYPE))

        if n_coords == 1 and batch_size > 1:
            # fetch neighbours with the same coordinates across the input batch
            inputs = tf.unstack(inputs)

            def _get_knot(bc):
                coord = [sc[c][i] for i, c in enumerate(bc)]
                coord = tf.stack(coord, axis=-1)
                batch_samples = [tf.gather_nd(img, coord) for img in inputs]
                batch_samples = tf.concat(batch_samples, axis=0)
                return batch_samples

        elif n_coords == batch_size:
            batch_ids = tf.reshape(
                tf.range(batch_size), [batch_size] + [1] * out_spatial_rank)
            batch_ids = tf.tile(batch_ids, [1] + out_spatial_size)

            def _get_knot(bc):
                coord = [batch_ids] + [sc[c][i] for i, c in enumerate(bc)]
                coord = tf.stack(coord, axis=-1)
                return tf.gather_nd(inputs, coord)
        else:
            raise NotImplementedError

        def _pyramid_combination(samples, w_0, w_1):
            # the case where n_coords = 1 and batch_size > 1 is handled by
            # shape broadcasting
            if len(w_0) == 1:
                return samples[0] * w_1[0] + samples[1] * w_0[0]
            f_0 = _pyramid_combination(samples[::2], w_0[:-1], w_1[:-1])
            f_1 = _pyramid_combination(samples[1::2], w_0[:-1], w_1[:-1])
            return f_0 * w_1[-1] + f_1 * w_0[-1]

        binary_neighbour_ids = _binary_neighbour_ids(in_spatial_rank)
        samples = [_get_knot(bc) for bc in binary_neighbour_ids]

        return _pyramid_combination(samples, weight_0, weight_1)

    def _resample_bspline(self, inputs, sample_coords):
        assert inputs.shape.is_fully_defined(), \
            "input shape should be fully defined for bspline interpolation"
        in_size = inputs.shape.as_list()
        batch_size = in_size[0]
        in_spatial_size = in_size[1:-1]
        in_spatial_rank = infer_spatial_rank(inputs)

        out_spatial_rank = infer_spatial_rank(sample_coords)
        if in_spatial_rank == 2:
            raise NotImplementedError(
                'bspline interpolation not implemented for 2d yet')
        assert batch_size == int(sample_coords.get_shape()[0])
        floor_coords = tf.floor(sample_coords)

        # Compute voxels to use for interpolation
        grid = tf.meshgrid([-1., 0., 1., 2.],
                           [-1., 0., 1., 2.],
                           [-1., 0., 1., 2.],
                           indexing='ij')
        offset_shape = [1, -1] + [1] * out_spatial_rank + [in_spatial_rank]
        offsets = tf.reshape(tf.stack(grid, 3), offset_shape)
        spatial_coords = offsets + tf.expand_dims(floor_coords, 1)
        spatial_coords = self.boundary_func(spatial_coords, in_spatial_size)
        spatial_coords = tf.cast(spatial_coords, COORDINATES_TYPE)
        knot_size = spatial_coords.shape.as_list()

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

    def _resample_inv_dst_weighting(self, inputs, sample_coords):
        # inverse distance weighting using 2^(sptial_rank) neighbours
        in_size = inputs.shape
        partial_shape = not in_size.is_fully_defined()
        try:
            batch_size = int(in_size[0])
            n_coords = int(sample_coords.shape[0])
            in_spatial_rank = infer_spatial_rank(inputs)
            in_spatial_size = \
                None if partial_shape else in_size.as_list()[1:-1]
        except (TypeError, AssertionError, ValueError):
            tf.logging.fatal('Unknown input shape, at least batch size '
                             'and rank of the inputs are required.')
            raise

        out_rank = len(sample_coords.get_shape())
        binary_neighbour_ids = _binary_neighbour_ids(in_spatial_rank)
        weight_id = [[[c, i] for i, c in enumerate(bc)]
                     for bc in binary_neighbour_ids]
        sample_coords_shape = [out_rank - 1, 0] + list(range(1, out_rank - 1))
        sample_coords = tf.transpose(sample_coords, sample_coords_shape)

        if partial_shape or in_spatial_size is None:
            all_coords_f = tf.stack(
                [tf.floor(sample_coords), tf.ceil(sample_coords)])
        else:
            # broadcasting input spatial size for boundary functions
            expanded_spatial_size = \
                [len(in_spatial_size)] + [1] * (out_rank - 1)
            b_size = tf.reshape(in_spatial_size, expanded_spatial_size)
            # find floor and ceil coordinates
            all_coords_f = tf.stack([
                self.boundary_func(tf.floor(sample_coords), b_size),
                self.boundary_func(tf.ceil(sample_coords), b_size)])

        # find N weights associated to each output point
        diff = tf.stack(
            [tf.squared_difference(sample_coords - EPS, all_coords_f[0]),
             tf.squared_difference(sample_coords + EPS, all_coords_f[1])])

        # gather_nd for both matrices, the same as:
        # point_weights = tf.gather_nd(diff, weight_id)
        # knots_id = tf.gather_nd(all_coords_f, weight_id)
        n_val = tf.gather_nd(
            tf.stack([diff, all_coords_f], axis=-1), weight_id)
        n_val = tf.unstack(n_val, axis=-1)
        point_weights, knots_id = n_val[0], n_val[1]

        # inverse distance weighting
        # sum_i (w_i*p_i/(sum_j w_j)) where w_i = 1/((p-p_i)^2)
        # point_weights has the shape:100
        # `[N, input_rank, b, sp_dim_0, ..., sp_dim_K]`
        # where:
        #  `N` is 2**source data spatial rank
        #  `b` is batch size,
        #  `sp_dim_0` is the output spatial output 0,
        #
        # `point_weights` represents (p - p_i)^2
        #      with i= 0...2**source_rank neighbours
        # (to do: these operations could be refactored as a resampling kernel)
        point_weights = tf.reduce_sum(point_weights, axis=1)
        # skip this as power = 2.0:
        # self.power = 2.0
        # point_weights = tf.pow(point_weights, self.power / 2.0)
        point_weights = tf.reciprocal(point_weights)
        point_weights = point_weights / tf.reduce_sum(point_weights, axis=0)

        # find N neighbours associated to each output point
        # knots_shape = tf.concat([[0], tf.range(2, out_rank + 1), [1]], 0)
        knots_shape = [0] + list(range(2, out_rank + 1)) + [1]
        knots_id = tf.cast(knots_id, COORDINATES_TYPE)
        knots_id = tf.transpose(knots_id, knots_shape)

        # get values of N neighbours
        batch_inputs = tf.unstack(inputs, axis=0)
        batch_knots = tf.unstack(knots_id, axis=1)
        if batch_size == n_coords:
            samples = [tf.gather_nd(img, knot)
                       for (img, knot) in zip(batch_inputs, batch_knots)]
        elif n_coords == 1 and batch_size > 1:
            samples = [tf.gather_nd(img, batch_knots[0])
                       for img in batch_inputs]
        else:
            raise NotImplementedError
        samples = tf.stack(samples, axis=1)
        # weighted average over N neighbours
        return tf.reduce_sum(
            samples * tf.expand_dims(point_weights, axis=-1), axis=0)


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
    # sample_coords = tf.cast(sample_coords, COORDINATES_TYPE)
    try:
        # if input_size is a numpy array
        input_size = tf.constant(input_size, dtype=sample_coords.dtype)
    except (TypeError, AttributeError):
        pass
    try:
        # if input_size is a TF Tensor
        input_size = tf.cast(input_size, dtype=sample_coords.dtype)
    except (TypeError, AttributeError):
        pass
    return sample_coords, input_size


def _binary_neighbour_ids(spatial_rank):
    """
    returns combinatorial binary indices::

        1-D: [[0], [1]]
        2-D: [[0, 0], [0, 1], [1, 0], [1, 1]]
        3-D: [[0, 0, 0], [0, 0, 1], [0, 1, 0],
              [0, 1, 1], [1, 0, 0], [1, 0, 1],
              [1, 1, 0], [1, 1, 1]]

    """
    return [[int(c) for c in format(i, '0%ib' % spatial_rank)]
            for i in range(2 ** spatial_rank)]


try:  # Some tf versions have this defined already
    @tf.RegisterGradient('FloorMod')
    def _floormod_grad(op, grad):
        return [None, None]
except:
    pass

SUPPORTED_INTERPOLATION = {'BSPLINE', 'LINEAR', 'NEAREST', 'IDW'}

SUPPORTED_BOUNDARY = {
    'ZERO': _boundary_replicate,
    'REPLICATE': _boundary_replicate,
    'CIRCULAR': _boundary_circular,
    'SYMMETRIC': _boundary_symmetric}

SUPPORTED_INPUT_DTYPE = {tf.float32}
