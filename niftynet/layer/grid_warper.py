# -*- coding: utf-8 -*-
# Copyright 2017 The Sonnet Authors. All Rights Reserved.
# Modifications copyright 2017 The NiftyNet Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Grid warper layer and utilities
adapted from
https://github.com/deepmind/sonnet/blob/v1.13/sonnet/python/modules/spatial_transformer.py
https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/v0.2.0.post1/niftynet/layer/spatial_transformer.py
"""
from __future__ import absolute_import, division, print_function

from itertools import chain

import numpy as np
import tensorflow as tf

from niftynet.layer.base_layer import Layer, LayerFromCallable, Invertible


class GridWarperLayer(Layer):
    """
    Grid warper interface class.

    An object implementing the `GridWarper` interface
    generates a reference grid of feature points at construction time,
    and warps it via a parametric transformation model,
    specified at run time by an input parameter Tensor.
    Grid warpers must then implement a `create_features` function
    used to generate the reference grid to be warped
    in the forward pass (according to a determined warping model).
    """

    def __init__(self,
                 source_shape,
                 output_shape,
                 coeff_shape,
                 name,
                 **kwargs):
        """
        Constructs a GridWarper module and
        initializes the source grid params.

        `source_shape` and `output_shape` defines the size of the source
        and output signal domains.

        For example,
        for an image of size `width=W` and `height=H`,
        `{source,output}_shape=[H, W]`;
        for a volume of size `width=W`, `height=H`
        and `depth=D`, `{source,output}_shape=[H, W, D]`.

        Args:
          source_shape: Iterable of integers determining
            the size of the source signal domain.
          output_shape: Iterable of integers determining
            the size of the destination resampled signal domain.
          coeff_shape: Shape of coefficients parameterizing the grid warp.
            For example, a 2D affine transformation will be defined by the [6]
            parameters populating the corresponding 2x3 affine matrix.
          name: Name of Module.
          **kwargs: Extra kwargs to be forwarded to
            the `create_features` function,
            instantiating the source grid parameters.

        Raises:
          Error: If `len(output_shape) > len(source_shape)`.
          TypeError: If `output_shape` and `source_shape`
            are not both iterable.
        """
        super(GridWarperLayer, self).__init__(name=name)

        self._source_shape = tuple(source_shape)
        self._output_shape = tuple(output_shape)
        if len(self._output_shape) > len(self._source_shape):
            tf.logging.fatal(
                'Output domain dimensionality (%s) must be equal or '
                'smaller than source domain dimensionality (%s)',
                len(self._output_shape), len(self._source_shape))
            raise ValueError

        self._coeff_shape = coeff_shape
        self._psi = self._create_features(**kwargs)

    def _create_features(self, **kwargs):
        """
        Precomputes features
        (e.g. sampling patterns, unconstrained feature matrices).
        """
        tf.logging.fatal('_create_features() should be implemented')
        raise NotImplementedError

    def layer_op(self, *args, **kwargs):
        tf.logging.fatal('layer_op() should be implemented to warp self._psi')
        raise NotImplementedError

    @property
    def coeff_shape(self):
        """Returns number of coefficients of warping function."""
        return self._coeff_shape

    @property
    def psi(self):
        """Returns a list of features used to compute the grid warp."""
        return self._psi

    @property
    def source_shape(self):
        """Returns a tuple containing the shape of the source signal."""
        return self._source_shape

    @property
    def output_shape(self):
        """Returns a tuple containing the shape of the output grid."""
        return self._output_shape


class AffineGridWarperLayer(GridWarperLayer, Invertible):
    """
    Affine Grid Warper class.

    The affine grid warper generates a reference grid of n-dimensional points
    and warps it via an affine transformation model determined by an input
    parameter Tensor. Some of the transformation parameters can be fixed at
    construction time via an `AffineWarpConstraints` object.
    """

    def __init__(self,
                 source_shape,
                 output_shape,
                 constraints=None,
                 name='affine_grid_warper'):
        """Constructs an AffineGridWarper.

        `source_shape` and `output_shape` are used to define shape of source
        and output signal domains, as opposed to the shape of the respective
        Tensors.
        For example, for an image of size `width=W` and `height=H`,
        `{source,output}_shape=[H, W]`;
        for a volume of size `width=W`, `height=H` and `depth=D`,
        `{source,output}_shape=[H, W, D]`.

        Args:
          source_shape: Iterable of integers determining shape of source
            signal domain.
          output_shape: Iterable of integers determining shape of destination
            resampled signal domain.
          constraints: Either a double list of shape `[N, N+1]`
            defining constraints
            on the entries of a matrix defining an affine transformation in N
            dimensions, or an `AffineWarpConstraints` object.
            If the double list is passed, a numeric value bakes
            in a constraint on the corresponding
            entry in the transformation matrix, whereas `None` implies that the
            corresponding entry will be specified at run time.
          name: Name of module.

        Raises:
          Error: If constraints fully define the affine transformation; or if
            input grid shape and constraints have different dimensionality.
          TypeError: If output_shape and source_shape are not both iterable.
        """
        self._source_shape = tuple(source_shape)
        self._output_shape = tuple(output_shape)
        num_dim = len(source_shape)
        if isinstance(constraints, AffineWarpConstraints):
            self._constraints = constraints
        elif constraints is None:
            self._constraints = AffineWarpConstraints.no_constraints(num_dim)
        else:
            self._constraints = AffineWarpConstraints(constraints=constraints)

        if self._constraints.num_free_params == 0:
            tf.logging.fatal('Transformation is fully constrained.')
            raise ValueError

        if self._constraints.num_dim != num_dim:
            tf.logging.fatal('Incompatible set of constraints provided: '
                             'input grid shape and constraints have different '
                             'dimensionality.')
            raise ValueError

        GridWarperLayer.__init__(
            self,
            source_shape=source_shape,
            output_shape=output_shape,
            coeff_shape=[6],
            name=name,
            constraints=self._constraints)

    def _create_features(self, constraints):
        """
        Creates all the matrices needed to compute the output warped grids.
        """
        affine_warp_constraints = constraints
        if not isinstance(affine_warp_constraints, AffineWarpConstraints):
            affine_warp_constraints = AffineWarpConstraints(constraints)
        mask = affine_warp_constraints.mask
        psi = _create_affine_features(output_shape=self._output_shape,
                                      source_shape=self._source_shape)
        scales = [1. for _ in self._source_shape]
        offsets = [0. for _ in self._source_shape]
        # Transforming a point x's i-th coordinate via an affine transformation
        # is performed via the following dot product:
        #
        #  x_i' = s_i * (T_i * x) + t_i                                    (1)
        #
        # where Ti is the i-th row of an affine matrix, and the scalars
        # s_i and t_i define a decentering and global scaling into
        # the source space.
        #
        # In the AffineGridWarper some of the entries of Ti are provided via the
        # input, some others are instead fixed, according to the constraints
        # assigned in the constructor.
        # In create_features the internal dot product (1) is accordingly
        # broken down into two parts:
        #
        # x_i' = Ti[uncon_i] * x[uncon_i, :] + offset(con_var)             (2)
        #
        # i.e. the sum of the dot product of the free parameters (coming
        # from the input) indexed by uncond_i and an offset obtained by
        # precomputing the fixed part of (1) according to the constraints.
        # This step is implemented by analyzing row by row
        # the constraints matrix and saving into a list
        # the x[uncon_i] and offset(con_var) data matrices
        # for each output dimension.
        features = []
        for row, scale in zip(mask, scales):
            x_i = np.array([x for x, is_active in zip(psi, row) if is_active])
            features.append(x_i * scale if len(x_i) else None)

        for row_i, row in enumerate(mask):
            x_i = None
            s = scales[row_i]
            for i, is_active in enumerate(row):
                if is_active:
                    continue

                # In principle a whole row of the affine matrix can be fully
                # constrained. In that case the corresponding dot product
                # between input parameters and grid coordinates doesn't need
                # to be implemented in the computation graph
                # since it can be precomputed.
                # When a whole row if constrained,
                # x_i - which is initialized to None - will still be None
                # at the end do the loop when it is appended
                # to the features list; this value is then used to
                # detect this setup in the build function where
                # the graph is assembled.
                if x_i is None:
                    x_i = np.array(psi[i]) * \
                          affine_warp_constraints[row_i][i] * s
                else:
                    x_i += np.array(psi[i]) * \
                           affine_warp_constraints[row_i][i] * s
            features.append(x_i)
        features += offsets
        return features

    @property
    def constraints(self):
        return self._constraints

    def layer_op(self, inputs):
        """Assembles the module network and adds it to the graph.

        The internal computation graph is assembled according to the set of
        constraints provided at construction time.

        inputs shape: batch_size x num_free_params

        Args:
          inputs: Tensor containing a batch of transformation parameters.

        Returns:
          A batch of warped grids.

        Raises:
          Error: If the input tensor size is not consistent
            with the constraints passed at construction time.
        """
        inputs = tf.to_float(inputs)

        input_shape = tf.shape(inputs)
        input_dtype = inputs.dtype.as_numpy_dtype
        batch_size = tf.expand_dims(input_shape[0], 0)
        number_of_params = inputs.shape[1]
        if number_of_params != self._constraints.num_free_params:
            tf.logging.fatal(
                'Input size is not consistent with constraint '
                'definition: %s parameters expected, %s provided.',
                self._constraints.num_free_params, number_of_params)
            raise ValueError
        num_output_dimensions = len(self._psi) // 3

        def get_input_slice(start, size):
            """
            Extracts a subset of columns from the input 2D Tensor.
            """
            rank = len(inputs.shape.as_list())
            return tf.slice(inputs,
                            begin=[0, start] + [0] * (rank - 2),
                            size=[-1, size] + [-1] * (rank - 2))

        warped_grid = []
        var_index_offset = 0
        number_of_points = np.prod(self._output_shape)
        for i in range(num_output_dimensions):
            if self._psi[i] is not None:
                # The i-th output dimension is not fully specified
                # by the constraints, the graph is setup to perform
                # matrix multiplication in batch mode.
                grid_coord = self._psi[i].astype(input_dtype)

                num_active_vars = self._psi[i].shape[0]
                active_vars = get_input_slice(var_index_offset, num_active_vars)
                warped_coord = tf.matmul(active_vars, grid_coord)
                warped_coord = tf.expand_dims(warped_coord, 1)
                var_index_offset += num_active_vars
                offset = self._psi[num_output_dimensions + i]
                if offset is not None:
                    offset = offset.astype(input_dtype)
                    # Some entries in the i-th row
                    # of the affine matrix were constrained
                    # and the corresponding matrix
                    # multiplications have been precomputed.
                    tiling_params = tf.concat(
                        [batch_size, tf.constant(1, shape=(1,)),
                         tf.ones_like(offset.shape)], 0)
                    offset = offset.reshape((1, 1) + offset.shape)
                    warped_coord += tf.tile(offset, tiling_params)

            else:
                # The i-th output dimension is fully specified
                # by the constraints, and the corresponding matrix
                # multiplications have been precomputed.
                warped_coord = \
                    self._psi[num_output_dimensions + i].astype(input_dtype)
                tiling_params = tf.concat(
                    [batch_size, tf.constant(1, shape=(1,)),
                     tf.ones_like(warped_coord.shape)], 0)
                warped_coord = warped_coord.reshape((1, 1) + warped_coord.shape)
                warped_coord = tf.tile(warped_coord, tiling_params)
            warped_coord = \
                warped_coord + self._psi[i + 2 * num_output_dimensions]
            # Need to help TF figuring out shape inference
            # since tiling information
            # is held in Tensors which are not known until run time.
            warped_coord.set_shape([None, 1, number_of_points])
            warped_grid.append(warped_coord)

        # Reshape all the warped coordinates tensors to
        # match the specified output
        # shape and concatenate into a single matrix.
        grid_shape = self._output_shape + (1,)
        warped_grid = [tf.reshape(grid, (-1,) + grid_shape)
                       for grid in warped_grid]
        return tf.concat(warped_grid, len(grid_shape))

    def inverse_op(self, name=None):
        """
        Returns a layer to compute inverse affine transforms.

          The function first assembles a network that
          given the constraints of the
          current AffineGridWarper and a set of input parameters,
          retrieves the coefficients of the corresponding inverse
          affine transform, then feeds its output into a new
          AffineGridWarper setup to correctly warp the `output`
          space into the `source` space.

        Args:
          name: Name of module implementing the inverse grid transformation.

        Returns:
          A `sonnet` module performing the inverse affine transform
          of a reference grid of points via an AffineGridWarper module.

        Raises:
          tf.errors.UnimplementedError: If the function is called on a non 2D
            instance of AffineGridWarper.
        """
        if self._coeff_shape != [6]:
            tf.logging.fatal('AffineGridWarper currently supports'
                             'inversion only for the 2D case.')
            raise NotImplementedError

        def _affine_grid_warper_inverse(inputs):
            """Assembles network to compute inverse affine transformation.

            Each `inputs` row potentially contains [a, b, tx, c, d, ty]
            corresponding to an affine matrix:

              A = [a, b, tx],
                  [c, d, ty]

            We want to generate a tensor containing the coefficients of the
            corresponding inverse affine transformation in a constraints-aware
            fashion.
            Calling M:

              M = [a, b]
                  [c, d]

            the affine matrix for the inverse transform is:

               A_in = [M^(-1), M^-1 * [-tx, -tx]^T]

            where

              M^(-1) = (ad - bc)^(-1) * [ d, -b]
                                        [-c,  a]

            Args:
              inputs: Tensor containing a batch of transformation parameters.

            Returns:
              A tensorflow graph performing the inverse affine transformation
              parametrized by the input coefficients.
            """
            batch_size = tf.expand_dims(tf.shape(inputs)[0], 0)
            constant_shape = tf.concat(
                [batch_size, tf.convert_to_tensor((1,))], 0)

            index = iter(range(6))

            def get_variable(constraint):
                if constraint is None:
                    i = next(index)
                    return inputs[:, i:i + 1]
                else:
                    return tf.fill(constant_shape,
                                   tf.constant(constraint, dtype=inputs.dtype))

            constraints = chain.from_iterable(self.constraints)
            a, b, tx, c, d, ty = (get_variable(constr) for constr in
                                  constraints)

            det = a * d - b * c
            a_inv = d / det
            b_inv = -b / det
            c_inv = -c / det
            d_inv = a / det

            m_inv = tf.reshape(
                tf.concat([a_inv, b_inv, c_inv, d_inv], 1), [-1, 2, 2])

            txy = tf.expand_dims(tf.concat([tx, ty], 1), 2)

            txy_inv = tf.reshape(tf.matmul(m_inv, txy), [-1, 2])
            tx_inv = txy_inv[:, 0:1]
            ty_inv = txy_inv[:, 1:2]

            inverse_gw_inputs = tf.concat(
                [a_inv, b_inv, -tx_inv, c_inv, d_inv, -ty_inv], 1)

            agw = AffineGridWarperLayer(self.output_shape, self.source_shape)

            return agw(inverse_gw_inputs)  # pylint: disable=not-callable

        if name is None:
            name = self.name + '_inverse'
        return LayerFromCallable(_affine_grid_warper_inverse, name=name)


class AffineWarpConstraints(object):
    """Affine warp constraints class.

    `AffineWarpConstraints` allow for
    very succinct definitions of constraints on
    the values of entries in affine transform matrices.
    """

    def __init__(self, constraints=((None,) * 3,) * 2):
        """Creates a constraint definition for an affine transformation.

        Args:
          constraints: A doubly-nested iterable of shape `[N, N+1]`
          defining constraints on the entries of a matrix that
          represents an affine transformation in `N` dimensions.
          A numeric value bakes in a constraint on the corresponding
          entry in the transformation matrix, whereas `None` implies that
          the corresponding entry will be specified at run time.

        Raises:
          TypeError: If `constraints` is not a nested iterable.
          ValueError: If the double iterable `constraints` has inconsistent
            dimensions.
        """
        try:
            self._constraints = tuple(tuple(x) for x in constraints)
        except TypeError:
            tf.logging.fatal('constraints must be a nested iterable.')
            raise TypeError

        # Number of rows
        self._num_dim = len(self._constraints)
        expected_num_cols = self._num_dim + 1
        if any(len(x) != expected_num_cols for x in self._constraints):
            tf.logging.fatal(
                'The input list must define a Nx(N+1) matrix of constraints.')
            raise ValueError

    def _calc_mask(self):
        """Computes a boolean mask from the user defined constraints."""
        mask = []
        for row in self._constraints:
            mask.append(tuple(x is None for x in row))
        return tuple(mask)

    def _calc_num_free_params(self):
        """Computes number of non constrained parameters."""
        return sum(row.count(None) for row in self._constraints)

    @property
    def num_free_params(self):
        return self._calc_num_free_params()

    @property
    def mask(self):
        return self._calc_mask()

    @property
    def constraints(self):
        return self._constraints

    @property
    def num_dim(self):
        return self._num_dim

    def __getitem__(self, i):
        """
        Returns the list of constraints
        for the i-th row of the affine matrix.
        """
        return self._constraints[i]

    def _combine(self, x, y):
        """
        Combines two constraints,
        raising an error if they are not compatible.
        """
        if x is None or y is None:
            return x or y
        if x != y:
            tf.logging.fatal('Incompatible set of constraints provided.')
            raise ValueError
        return x

    def __and__(self, rhs):
        """Combines two sets of constraints into a coherent single set."""
        return self.combine_with(rhs)

    def combine_with(self, additional_constraints):
        """Combines two sets of constraints into a coherent single set."""
        x = additional_constraints
        if not isinstance(additional_constraints, AffineWarpConstraints):
            x = AffineWarpConstraints(additional_constraints)
        new_constraints = []
        for left, right in zip(self._constraints, x.constraints):
            new_constraints.append(
                [self._combine(x, y) for x, y in zip(left, right)])
        return AffineWarpConstraints(new_constraints)

    # Collection of utilities to initialize an AffineGridWarper in 2D and 3D.
    @classmethod
    def no_constraints(cls, num_dim=2):
        """
        Empty set of constraints for a num_dim affine transform.
        """
        return cls(((None,) * (num_dim + 1),) * num_dim)

    @classmethod
    def translation_2d(cls, x=None, y=None):
        """
        Assign constraints on translation components of
        affine transform in 2d.
        """
        return cls([[None, None, x],
                    [None, None, y]])

    @classmethod
    def translation_3d(cls, x=None, y=None, z=None):
        """
        Assign constraints on translation components of
        affine transform in 3d.
        """
        return cls([[None, None, None, x],
                    [None, None, None, y],
                    [None, None, None, z]])

    @classmethod
    def scale_2d(cls, x=None, y=None):
        """
        Assigns constraints on scaling components of
        affine transform in 2d.
        """
        return cls([[x, None, None],
                    [None, y, None]])

    @classmethod
    def scale_3d(cls, x=None, y=None, z=None):
        """
        Assigns constraints on scaling components of
        affine transform in 3d.
        """
        return cls([[x, None, None, None],
                    [None, y, None, None],
                    [None, None, z, None]])

    @classmethod
    def shear_2d(cls, x=None, y=None):
        """
        Assigns constraints on shear components of
        affine transform in 2d.
        """
        return cls([[None, x, None],
                    [y, None, None]])

    @classmethod
    def no_shear_2d(cls):
        return cls.shear_2d(x=0, y=0)

    @classmethod
    def no_shear_3d(cls):
        """
        Assigns constraints on shear components of
        affine transform in 3d.
        """
        return cls([[None, 0, 0, None],
                    [0, None, 0, None],
                    [0, 0, None, None]])


def _create_affine_features(output_shape, source_shape):
    """
    Generates n-dimensional homogeneous coordinates
    for a given grid definition.
    `source_shape` and `output_shape` are used to
    define the size of the source and output signal domains.

    For example,
    for an image of size `width=W` and `height=H`,
    `{source,output}_shape=[H, W]`;
    for a volume of size `width=W`, `height=H` and `depth=D`,
    `{source,output}_shape=[H, W, D]`.

    Note returning in Matrix indexing 'ij'

    Args:
      output_shape: Iterable of integers determining
        the shape of the grid to be warped.
      source_shape: Iterable of integers determining
        the domain of the signal to be resampled.
    Returns:
      List of flattened numpy arrays of coordinates
      When the dimensionality of `output_shape` is smaller that that of
      `source_shape` the last rows before [1, ..., 1] will be filled with 0.
    """
    dim_gap = len(source_shape) - len(output_shape)
    embedded_output_shape = list(output_shape) + [1] * dim_gap
    ranges = [np.arange(dim, dtype=np.float32)
              for dim in embedded_output_shape]
    ranges.append(np.array([1.0]))
    return [x.ravel() for x in np.meshgrid(*ranges, indexing='ij')]
