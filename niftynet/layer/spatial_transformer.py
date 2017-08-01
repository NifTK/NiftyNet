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

""""Implementation of Spatial Transformer networks core components."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Dependency imports
import numpy as np
import tensorflow as tf
from niftynet.layer.base_layer import Layer
from niftynet.layer.base_layer import LayerFromCallable
from niftynet.layer import layer_util

import abc
from itertools import chain
from six.moves import xrange  # pylint: disable=redefined-builtin
from niftynet.utilities.misc_common import look_up_operations

SUPPORTED_INTERPOLATION={'BSPLINE','LINEAR','NEAREST'}
SUPPORTED_BOUNDARY={'ZERO','REPLICATE','CIRCULAR','SYMMETRIC'}

class ResamplerLayer(Layer):
  """ Resampler  class
  
  Takes an input tensor and
  a sampling definition (e.g. a list of grid points) at run-time and returns 
  one or more values for each grid location
  """
  def __init__(self, interpolation='LINEAR', boundary='REPLICATE',name='resampler'):
    super(ResamplerLayer, self).__init__(name=name)
    self.interpolation = look_up_operations(interpolation.upper(), SUPPORTED_INTERPOLATION)
    self.boundary = look_up_operations(boundary.upper(), SUPPORTED_BOUNDARY)
    if self.boundary == 'ZERO' and self.interpolation in ['BSPLINE', 'NEAREST']:
      raise NotImplementedError('Zero padding is only supported for linear interpolation currently')
    self.boundary_func_ = {'ZERO': self.boundary_replicate,  # zero is replicate with special edge handling (hack)
                           'REPLICATE':self.boundary_replicate,
                           'CIRCULAR':self.boundary_circular,
                           'SYMMETRIC':self.boundary_symmetric}[self.boundary]
    self.resample_func_ = {'LINEAR':self.resample_linear,
                           'BSPLINE':self.resample_bspline,
                           'NEAREST':self.resample_nearest}[self.interpolation]

  def boundary_replicate(self,sample_coords,input_size):
    return tf.maximum(tf.minimum(sample_coords,input_size-1),0)
  def boundary_circular(self,sample_coords,input_size):
    return tf.mod(tf.mod(sample_coords,input_size)+input_size,input_size)
  def boundary_symmetric(self,sample_coords,input_size):
    circularSize = input_size+input_size-2
    return (input_size-1)-tf.abs((input_size-1)-tf.mod(tf.mod(sample_coords,circularSize)+circularSize,circularSize))

  def resample_bspline(self,inputs,sample_coords):
    input_size=tf.reshape(inputs.get_shape().as_list()[1:-1],[1]*(len(sample_coords.get_shape().as_list())-1)+[-1])
    spatial_rank = layer_util.infer_spatial_rank(inputs)
    batch_size=sample_coords.get_shape().as_list()[0]
    grid_shape = sample_coords.get_shape().as_list()[1:-1]
    if spatial_rank==2:
      raise NotImplementedError('bspline interpolation not implemented for 2d yet')
    index_voxel_coords = tf.floor(sample_coords)
    # Compute voxels to use for interpolation
    grid=tf.meshgrid(list(range(-1,3)),list(range(-1,3)),list(range(-1,3)), indexing='ij')
    offsets = tf.reshape(tf.stack(grid,3),[1,4**spatial_rank]+[1]*len(grid_shape)+[spatial_rank])
    preboundary_spatial_coords = offsets+tf.expand_dims(tf.cast(index_voxel_coords,tf.int32),1)
    spatial_coords = self.boundary_func_(preboundary_spatial_coords,input_size)
    sz=spatial_coords.get_shape().as_list()
    # Compute weights for each voxel
    build_coefficient = lambda u,d: tf.concat([tf.pow(1-u,3),
                                             3*tf.pow(u,3) - 6*tf.pow(u,2) + 4,
                                            -3*tf.pow(u,3) + 3*tf.pow(u,2) + 3*u + 1,
                                               tf.pow(u,3)],d)/6

    weight=tf.reshape(sample_coords-index_voxel_coords,[batch_size,-1,3])
    Bu=build_coefficient(tf.reshape(weight[:,:,0],[batch_size,1,1,1,-1]),1)
    Bv=build_coefficient(tf.reshape(weight[:,:,1],[batch_size,1,1,1,-1]),2)
    Bw=build_coefficient(tf.reshape(weight[:,:,2],[batch_size,1,1,1,-1]),3)
    all_weights=tf.reshape(Bu*Bv*Bw,[batch_size] +sz[1:-1]+[1])
    # Gather voxel values and compute weighted sum
    batch_coords = tf.tile(tf.reshape(tf.range(sz[0]),[sz[0]]+[1]*(len(sz)-1)),[1]+sz[1:-1]+[1])
    raw_samples = tf.gather_nd(inputs,tf.concat([batch_coords,spatial_coords],-1))
    return tf.reduce_sum(all_weights*raw_samples,reduction_indices=1)

  def resample_linear(self,inputs,sample_coords):
    input_size = inputs.get_shape().as_list()[1:-1]
    spatial_rank = layer_util.infer_spatial_rank(inputs)

    xy=tf.unstack(sample_coords,axis=len(sample_coords.get_shape())-1)
    index_voxel_coords = [tf.floor(x) for x in xy]
    spatial_coords=[self.boundary_func_(tf.cast(x,tf.int32), input_size[idx])
                    for idx,x in enumerate(index_voxel_coords)]
    spatial_coords_plus1=[self.boundary_func_(tf.cast(x+1.,tf.int32), input_size[idx])
                          for idx,x in enumerate(index_voxel_coords)]
    if self.boundary == 'ZERO':  #
      weight = [tf.expand_dims(x - tf.cast(i, tf.float32), -1) for x, i in zip(xy, spatial_coords)]
      weight_c = [tf.expand_dims(tf.cast(i, tf.float32) - x, -1) for x, i in zip(xy, spatial_coords_plus1)]
    else:
      weight = [tf.expand_dims(x - i, -1) for x, i in zip(xy, index_voxel_coords)]
      weight_c = [1. - w for w in weight]
    sz = spatial_coords[0].get_shape().as_list()
    batch_coords = tf.tile(tf.reshape(tf.range(sz[0]), [sz[0]] + [1] * (len(sz) - 1)), [1] + sz[1:] )
    sc=(spatial_coords,spatial_coords_plus1)
    binary_codes = [[int(c) for c in format(i,'0%ib'%spatial_rank)] for i in range(2**spatial_rank)]
    make_sample = lambda bc: tf.gather_nd(inputs, tf.stack([batch_coords] + [sc[c][i] for i,c in enumerate(bc)] , -1))
    samples = [make_sample(bc) for bc in binary_codes]
    def pyramid_combination(samples,weight,weight_c):
      if len(weight)==1:
        return samples[0]*weight_c[0]+samples[1]*weight[0]
      else:
        return pyramid_combination(samples[::2], weight[:-1], weight_c[:-1]) * weight_c[-1] + \
               pyramid_combination(samples[1::2], weight[:-1], weight_c[:-1]) * weight[-1]
    return pyramid_combination(samples, weight, weight_c)

  def resample_nearest(self,inputs,sample_coords):
    input_size=tf.reshape(inputs.get_shape().as_list()[1:-1],[1]*(len(sample_coords.get_shape().as_list())-1)+[-1]  )
    spatial_rank = layer_util.infer_spatial_rank(inputs)
    spatial_coords = self.boundary_func_(tf.cast(tf.round(sample_coords),tf.int32),input_size);
    sz=spatial_coords.get_shape().as_list()
    batch_coords = tf.tile(tf.reshape(tf.range(sz[0]),[sz[0]]+[1]*(len(sz)-1)),[1]+sz[1:-1]+[1])
    return tf.gather_nd(inputs,tf.concat([batch_coords,spatial_coords],-1))
    
  def layer_op(self, inputs, sample_coords):
    return self.resample_func_(inputs,sample_coords)


class GridWarperLayer(Layer):
  """Grid warper interface class.

  An object implementing the `GridWarper` interface generates a reference grid
  of feature points at construction time, and warps it via a parametric
  transformation model, specified at run time by an input parameter Tensor.
  Grid warpers must then implement a `create_features` function used to generate
  the reference grid to be warped in the forward pass (according to a determined
  warping model).
  """

  def __init__(self, source_shape, output_shape, coeff_shape, name, **kwargs):
    """Constructs a GridWarper module and initializes the source grid params.

    `source_shape` and `output_shape` are used to define the size of the source
    and output signal domains, as opposed to the shape of the respective
    Tensors. For example, for an image of size `width=W` and `height=H`,
    `{source,output}_shape=[H, W]`; for a volume of size `width=W`, `height=H`
    and `depth=D`, `{source,output}_shape=[H, W, D]`.

    Args:
      source_shape: Iterable of integers determining the size of the source
        signal domain.
      output_shape: Iterable of integers determining the size of the destination
        resampled signal domain.
      coeff_shape: Shape of coefficients parametrizing the grid warp.
        For example, a 2D affine transformation will be defined by the [6]
        parameters populating the corresponding 2x3 affine matrix.
      name: Name of Module.
      **kwargs: Extra kwargs to be forwarded to the `create_features` function,
        instantiating the source grid parameters.

    Raises:
      Error: If `len(output_shape) > len(source_shape)`.
      TypeError: If `output_shape` and `source_shape` are not both iterable.
    """
    super(GridWarperLayer, self).__init__(name=name)

    self._source_shape = tuple(source_shape)
    self._output_shape = tuple(output_shape)
    if len(self._output_shape) > len(self._source_shape):
      raise ValueError('Output domain dimensionality ({}) must be equal or '
                       'smaller than source domain dimensionality ({})'
                       .format(len(self._output_shape),
                               len(self._source_shape)))

    self._coeff_shape = coeff_shape
    self._psi = self._create_features(**kwargs)

  @abc.abstractmethod
  def _create_features(self, **kwargs):
    """Precomputes features (e.g. sampling patterns, unconstrained feature matrices)."""
    pass

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


class BSplineFieldImageGridWarperLayer(GridWarperLayer):
  """ The fast BSpline Grid Warper defines a grid based on
      sampling coordinate values from a spatially varying displacement
      field  (passed as a tensor input) along a regular cartesian grid 
      pattern aligned with the field. Specifically,
  this class defines a grid based on BSpline smoothing, as described by Rueckert et al.
      To ensure that it can be done efficiently, several assumptions are made:
      1) The grid is a cartesian grid aligned with the field.
      2) Knots occur every M,N,O grid points (in X,Y,Z) This allows the 
         smoothing to be represented as a 4x4x4 convolutional kernel with MxNxO channels
  """
  def __init__(self,
               source_shape,
               output_shape,
               knot_spacing,
               name='interpolated_spline_grid_warper_layer'):
    """Constructs an BSplineFieldImageGridWarperLayer.
    Args:
      source_shape: Iterable of integers determining the size of the source
        signal domain.
      output_shape: Iterable of integers determining the size of the destination
        resampled signal domain.
      knot_spacing: List of intervals (in voxels) in each dimension where 
        displacements are defined in the field.
      interpolation: type of interpolation as used by tf.image.resize_images
      name: Name of Module."""
    coeff_shape=[4+(n-1)//k for n,k in zip(output_shape,knot_spacing)]
    self._knot_spacing=knot_spacing
    super(BSplineFieldImageGridWarperLayer, self).__init__(source_shape=source_shape,
                                           output_shape=output_shape,
                                           coeff_shape=coeff_shape,
                                           name=name)  
  def _create_features(self):
    """ Creates the convolutional kernel"""
    build_coefficient = lambda u,d: np.reshape(np.stack([(np.power(1-u,3))/6,
                                     (3*np.power(u,3) - 6*np.power(u,2) + 4)/6,
                                     (-3*np.power(u,3) + 3*np.power(u,2) + 3*u + 1)/6,
                                      np.power(u,3)/6],0),np.roll([4,1,1,len(u),1,1],d))
    coeffs = [build_coefficient(np.arange(k)/k,d) for d,k in enumerate(self._knot_spacing)]
    kernels = tf.constant(np.reshape(np.prod(coeffs),[4,4,4,1,-1]),dtype=tf.float32)
    return kernels
  def layer_op(self,field):
    batch_size=int(field.get_shape().as_list()[0])
    spatial_rank = int(field.get_shape().as_list()[-1])
    resampled_list=[tf.nn.conv3d(field[:, :, :, :, d:d + 1], self._psi, strides=[1]*5, padding='VALID')
                    for d in [0, 1, 2]]
    resampled=tf.stack(resampled_list,5)
    permuted_shape=[batch_size]+[f-3 for f in self._coeff_shape]+self._knot_spacing+[spatial_rank]
    print(permuted_shape)
    permuted=tf.transpose(tf.reshape(resampled,permuted_shape),[0,1,4,2,5,3,6,7])
    valid_size=[(f-3)*k for f,k in zip(self._coeff_shape,self._knot_spacing)]
    reshaped=tf.reshape(permuted,[batch_size]+valid_size+[spatial_rank])
    cropped = reshaped[:,:self._output_shape[0],:self._output_shape[1],:self._output_shape[2],:]
    return cropped

class RescaledFieldImageGridWarperLayer(GridWarperLayer):
  """ The rescaled field grid warper defines a grid based on
      sampling coordinate values from a spatially varying displacement
      field  (passed as a tensor input) along a regular cartesian grid 
      pattern aligned with the field. Specifically, this class defines
      a grid by resampling the field (using tf.rescale_images with 
      align_corners=False) to the output_shape.
  """
  def __init__(self,
               source_shape,
               output_shape,
               coeff_shape,
               interpolation=tf.image.ResizeMethod.BICUBIC,
               name='rescaling_interpolated_spline_grid_warper_layer'):
    """ Constructs an RescaledFieldImageGridWarperLayer.
    Args:
      source_shape: Iterable of integers determining the size of the source
        signal domain.
      output_shape: Iterable of integers determining the size of the destination
        resampled signal domain.
      coeff_shape: Shape of displacement field.
      interpolation: type of interpolation as used by tf.image.resize_images
      name: Name of Module.

    """
    self._interpolation=interpolation
    if self._interpolation=='LINEAR':
      self._interpolation=tf.image.ResizeMethod.BILINEAR
    elif self._interpolation=='CUBIC':
      self._interpolation=tf.image.ResizeMethod.BICUBIC
    
    super(RescaledFieldImageGridWarperLayer, self).__init__(source_shape=source_shape,
                                           output_shape=output_shape,
                                           coeff_shape=coeff_shape,
                                           field_interpretation=field_interpretation,
                                           name=name)
  def layer_op(self,field):
    input_shape = tf.shape(field)
    input_dtype = field.dtype.as_numpy_dtype
    batch_size = int(field.get_shape()[0])
    reshaped_field=tf.reshape(field, [batch_size, self._coeff_shape[0], self._coeff_shape[1], -1])
    coords_intermediate = tf.image.resize_images(reshaped_field,self._output_shape[0:2],
                                                 self._interpolation,align_corners=False)
    sz_xy_z1=[batch_size,self._output_shape[0]*self._output_shape[1],self._coeff_shape[2],-1]
    tmp=tf.reshape(coords_intermediate,sz_xy_z1)
    final_sz=[batch_size]+list(self._output_shape)+[-1]
    sz_xy_z2=[self._output_shape[0]*self._output_shape[1],self._output_shape[2]]
    coords=tf.reshape(tf.image.resize_images(tmp,sz_xy_z2,self._interpolation,align_corners=False),final_sz)
    return coords


class ResampledFieldGridWarperLayer(GridWarperLayer):
  """ The resampled field grid warper defines a grid based on
      sampling coordinate values from a spatially varying displacement
      field  (passed as a tensor input) along an affine grid pattern 
      in the field. 
      This enables grids representing small patches of a larger transform, 
      as well as the composition of multiple transforms before sampling.
  """
  def __init__(self,
               source_shape,
               output_shape,
               coeff_shape,
               field_transform=None,
               resampler=None,
               name='resampling_interpolated_spline_grid_warper'):
    """Constructs an ResampledFieldingGridWarperLayer.
    Args:
      source_shape: Iterable of integers determining the size of the source
        signal domain.
      output_shape: Iterable of integers determining the size of the destination
        resampled signal domain.
      coeff_shape: Shape of displacement field.
      interpolation: type of interpolation as used by tf.image.resize_images
      name: Name of Module.
      field_transform: an object defining the spatial relationship between the 
        output_grid and the field. 
        batch_size x4x4 tensor: per-image transform matrix from output coords to field coords
        None (default):         corners of output map to corners of field with an allowance for
                                  interpolation (1 for bspline, 0 for linear)
      resampler: a ResamplerLayer used to interpolate the 
        deformation field
      name: Name of module.

    Raises:
      TypeError: If output_shape and source_shape are not both iterable.
    """
    if resampler==None:
      self._resampler=ResamplerLayer(interpolation='LINEAR',boundary='REPLICATE')
      self._interpolation = 'LINEAR'
    else:
      self._resampler=resampler
      self._interpolation = self._resampler.interpolation
    
    self._field_transform = field_transform
    
    super(ResampledFieldGridWarperLayer, self).__init__(source_shape=source_shape,
                                           output_shape=output_shape,
                                           coeff_shape=coeff_shape,
                                           name=name)

  def _create_features(self):
    """Creates the coordinates for resampling. If field_transform is
    None, these are constant and are created in field space; otherwise,
    the final coordinates will be transformed by an input tensor
    representing a transform from output coordinates to field
    coordinates, so they are created are created in output coordinate
    space
    """
    embedded_output_shape = list(self._output_shape)+[1]*(len(self._source_shape) - len(self._output_shape))
    embedded_coeff_shape = list(self._coeff_shape)+[1]*(len(self._source_shape) - len(self._output_shape))
    if self._field_transform==None and self._interpolation == 'BSPLINE':
      range_func= lambda f,x: tf.linspace(1.,f-2.,x)
    elif self._field_transform==None and self._interpolation != 'BSPLINE':
      range_func= lambda f,x: tf.linspace(0.,f-1.,x)
    else:
      range_func= lambda f,x: np.arange(x,dtype=np.float32)
      embedded_output_shape+=[1] # make homogeneous
      embedded_coeff_shape+=[1]
    ranges = [range_func(f,x) for f,x in zip(embedded_coeff_shape,embedded_output_shape)]
    coords= tf.stack([tf.reshape(x,[1,-1]) for x in tf.meshgrid(*ranges, indexing='ij')],2)
    return coords

  def layer_op(self, field):
    """Assembles the module network and adds it to the graph.

    The internal computation graph is assembled according to the set of
    constraints provided at construction time.

    Args:
      field: Tensor containing a batch of transformation parameters.

    Returns:
      A batch of warped grids.

    Raises:
      Error: If the input tensor size is not consistent with the constraints
        passed at construction time.
    """
    input_shape = tf.shape(field)
    input_dtype = field.dtype.as_numpy_dtype
    batch_size = int(field.get_shape()[0])
    
    # transform grid into field coordinate space if necessary
    if self._field_transform==None:
      coords=self._psi
    else:
      coords = tf.matmul(self._psi,self._field_transform[:,:,1:3])
    # resample
    coords = tf.reshape(tf.tile(coords,[batch_size,1,1]),[-1]+list(self._output_shape)+[len(self._source_shape)])
    resampled_coords = self._resampler(field, coords)
    return resampled_coords

    
def _create_affine_features(output_shape, source_shape):
  """Generates n-dimensional homogenous coordinates for a given grid definition.
    `source_shape` and `output_shape` are used to define the size of the source
  and output signal domains, as opposed to the shape of the respective
  Tensors. For example, for an image of size `width=W` and `height=H`,
  `{source,output}_shape=[H, W]`; for a volume of size `width=W`, `height=H`
  and `depth=D`, `{source,output}_shape=[H, W, D]`.

  Args:
    output_shape: Iterable of integers determining the shape of the grid to be
      warped.
   source_shape: Iterable of integers determining the domain of the signal to be
     resampled.
  Returns:
    List of flattened numpy arrays of coordinates

  """
  embedded_output_shape = list(output_shape)+[1]*(len(source_shape) - len(output_shape))
  ranges = [np.arange(x,dtype=np.float32) for x in embedded_output_shape]+[np.array([1])]
  return [x.reshape(-1) for x in np.meshgrid(*ranges, indexing='ij')]

class AffineGridWarperLayer(GridWarperLayer):
  """Affine Grid Warper class.

  The affine grid warper generates a reference grid of n-dimensional points
  and warps it via an affine transormation model determined by an input
  parameter Tensor. Some of the transformation parameters can be fixed at
  construction time via an `AffineWarpConstraints` object.
  """

  def __init__(self,
               source_shape,
               output_shape,
               constraints=None,
               name='affine_grid_warper'):
    """Constructs an AffineGridWarper.

    `source_shape` and `output_shape` are used to define the size of the source
    and output signal domains, as opposed to the shape of the respective
    Tensors. For example, for an image of size `width=W` and `height=H`,
    `{source,output}_shape=[H, W]`; for a volume of size `width=W`, `height=H`
    and `depth=D`, `{source,output}_shape=[H, W, D]`.

    Args:
      source_shape: Iterable of integers determining the size of the source
        signal domain.
      output_shape: Iterable of integers determining the size of the destination
        resampled signal domain.
      constraints: Either a double list of shape `[N, N+1]` defining constraints
        on the entries of a matrix defining an affine transformation in N
        dimensions, or an `AffineWarpConstraints` object. If the double list is
        passed, a numeric value bakes in a constraint on the corresponding
        entry in the tranformation matrix, whereas `None` implies that the
        corresponding entry will be specified at run time.
      name: Name of module.

    Raises:
      Error: If constraints fully define the affine transformation; or if
        input grid shape and contraints have different dimensionality.
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
      raise ValueError('Transformation is fully constrained.')

    if self._constraints.num_dim != num_dim:
      raise ValueError('Incompatible set of constraints provided: '
                       'input grid shape and constraints have different '
                       'dimensionality.')

    super(AffineGridWarperLayer, self).__init__(source_shape=source_shape,
                                           output_shape=output_shape,
                                           coeff_shape=[6],
                                           name=name,
                                           constraints=self._constraints)

  def _create_features(self, constraints):
    """Creates all the matrices needed to compute the output warped grids."""
    affine_warp_constraints = constraints
    if not isinstance(affine_warp_constraints, AffineWarpConstraints):
      affine_warp_constraints = AffineWarpConstraints(affine_warp_constraints)
    mask = affine_warp_constraints.mask
    psi = _create_affine_features(output_shape=self._output_shape,
                                  source_shape=self._source_shape)
    scales = [1. for x in self._source_shape]
    offsets = [0. for x in scales]
    # Transforming a point x's i-th coordinate via an affine transformation
    # is performed via the following dot product:
    #
    #  x_i' = s_i * (T_i * x) + t_i                                          (1)
    #
    # where Ti is the i-th row of an affine matrix, and the scalars s_i and t_i
    # define a decentering and global scaling into the source space.
    # In the AffineGridWarper some of the entries of Ti are provided via the
    # input, some others are instead fixed, according to the constraints
    # assigned in the constructor.
    # In create_features the internal dot product (1) is accordingly broken down
    # into two parts:
    #
    # x_i' = Ti[uncon_i] * x[uncon_i, :] + offset(con_var)                   (2)
    #
    # i.e. the sum of the dot product of the free parameters (coming
    # from the input) indexed by uncond_i and an offset obtained by
    # precomputing the fixed part of (1) according to the constraints.
    # This step is implemented by analyzing row by row the constraints matrix
    # and saving into a list the x[uncon_i] and offset(con_var) data matrices
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
        # constrained. In that case the corresponding dot product between input
        # parameters and grid coordinates doesn't need to be implemented in the
        # computation graph since it can be precomputed.
        # When a whole row if constrained, x_i - which is initialized to
        # None - will still be None at the end do the loop when it is appended
        # to the features list; this value is then used to detect this setup
        # in the build function where the graph is assembled.
        if x_i is None:
          x_i = np.array(psi[i]) * affine_warp_constraints[row_i][i] * s
        else:
          x_i += np.array(psi[i]) * affine_warp_constraints[row_i][i] * s
      features.append(x_i)

    features += offsets
    return features

  def layer_op(self, inputs):
    """Assembles the module network and adds it to the graph.

    The internal computation graph is assembled according to the set of
    constraints provided at construction time.

    Args:
      inputs: Tensor containing a batch of transformation parameters.

    Returns:
      A batch of warped grids.

    Raises:
      Error: If the input tensor size is not consistent with the constraints
        passed at construction time.
    """
    input_shape = tf.shape(inputs)
    input_dtype = inputs.dtype.as_numpy_dtype
    batch_size = tf.expand_dims(input_shape[0], 0)
    number_of_params = inputs.get_shape()[1]
    if number_of_params != self._constraints.num_free_params:
      raise ValueError('Input size is not consistent with constraint '
                       'definition: {} parameters expected, {} provided.'
                       .format(self._constraints.num_free_params,
                               number_of_params))
    num_output_dimensions = len(self._psi) // 3
    def get_input_slice(start, size):
      """Extracts a subset of columns from the input 2D Tensor."""
      rank = len(inputs.get_shape().as_list())
      return tf.slice(inputs,begin=[0,start]+[0]*(rank-2),size=[-1,size]+[-1]*(rank-2))
    warped_grid = []
    var_index_offset = 0
    number_of_points = np.prod(self._output_shape)
    for i in xrange(num_output_dimensions):
      if self._psi[i] is not None:
        # The i-th output dimension is not fully specified by the constraints,
        # the graph is setup to perform matrix multiplication in batch mode.
        grid_coord = self._psi[i].astype(input_dtype)

        num_active_vars = self._psi[i].shape[0]
        active_vars = get_input_slice(var_index_offset, num_active_vars)
        warped_coord = tf.matmul(active_vars, grid_coord)
        warped_coord = tf.expand_dims(warped_coord, 1)
        var_index_offset += num_active_vars
        offset = self._psi[num_output_dimensions + i]
        if offset is not None:
          offset = offset.astype(input_dtype)
          # Some entries in the i-th row of the affine matrix were constrained
          # and the corresponding matrix multiplications have been precomputed.
          tiling_params = tf.concat(
              [
                  batch_size, tf.constant(
                      1, shape=(1,)), tf.ones_like(offset.shape)
              ],
              0)
          offset = offset.reshape((1, 1) + offset.shape)
          warped_coord += tf.tile(offset, tiling_params)

      else:
        # The i-th output dimension is fully specified by the constraints, and
        # the corresponding matrix multiplications have been precomputed.
        warped_coord = self._psi[num_output_dimensions + i].astype(input_dtype)
        tiling_params = tf.concat(
            [
                batch_size, tf.constant(
                    1, shape=(1,)), tf.ones_like(warped_coord.shape)
            ],
            0)
        warped_coord = warped_coord.reshape((1, 1) + warped_coord.shape)
        warped_coord = tf.tile(warped_coord, tiling_params)

      warped_coord += self._psi[i + 2 * num_output_dimensions]
      # Need to help TF figuring out shape inference since tiling information
      # is held in Tensors which are not known until run time.
      warped_coord.set_shape([None, 1, number_of_points])
      warped_grid.append(warped_coord)

    # Reshape all the warped coordinates tensors to match the specified output
    # shape and concatenate  into a single matrix.
    grid_shape = self._output_shape + (1,)
    warped_grid = [tf.reshape(grid,(-1,)+grid_shape) for grid in warped_grid]
    return tf.concat(warped_grid, len(grid_shape))

  @property
  def constraints(self):
    return self._constraints

  def inverse(self, name=None):
    """Returns a `sonnet` module to compute inverse affine transforms.

      The function first assembles a network that given the constraints of the
      current AffineGridWarper and a set of input parameters, retrieves the
      coefficients of the corresponding inverse affine transform, then feeds its
      output into a new AffineGridWarper setup to correctly warp the `output`
      space into the `source` space.

    Args:
      name: Name of module implementing the inverse grid transformation.

    Returns:
      A `sonnet` module performing the inverse affine transform of a reference
      grid of points via an AffineGridWarper module.

    Raises:
      tf.errors.UnimplementedError: If the function is called on a non 2D
        instance of AffineGridWarper.
    """
    if self._coeff_shape != [6]:
      raise tf.errors.UnimplementedError('AffineGridWarper currently supports'
                                         'inversion only for the 2D case.')
    def _affine_grid_warper_inverse(inputs):
      """Assembles network to compute inverse affine transformation.

      Each `inputs` row potentailly contains [a, b, tx, c, d, ty]
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
      constant_shape = tf.concat([batch_size, tf.convert_to_tensor((1,))], 0)

      index = iter(range(6))
      def get_variable(constraint):
        if constraint is None:
          i = next(index)
          return inputs[:, i:i+1]
        else:
          return tf.fill(constant_shape, tf.constant(constraint,
                                                     dtype=inputs.dtype))

      constraints = chain.from_iterable(self.constraints)
      a, b, tx, c, d, ty = (get_variable(constr) for constr in constraints)

      det = a * d - b * c
      a_inv = d / det
      b_inv = -b / det
      c_inv = -c / det
      d_inv = a / det

      m_inv = tf.reshape(tf.concat([a_inv, b_inv, c_inv, d_inv], 1),[-1, 2, 2])

      txy = tf.expand_dims(tf.concat([tx, ty], 1), 2)

      txy_inv = tf.reshape(tf.matmul(m_inv, txy),[-1,2])
      tx_inv = txy_inv[:, 0:1]
      ty_inv = txy_inv[:, 1:2]

      inverse_gw_inputs = tf.concat(
          [a_inv, b_inv, -tx_inv, c_inv, d_inv, -ty_inv], 1)

      agw = AffineGridWarper(self.output_shape,
                             self.source_shape)


      return agw(inverse_gw_inputs)  # pylint: disable=not-callable

    if name is None:
      name = self.module_name + '_inverse'
    return LayerFromCallable(_affine_grid_warper_inverse, name=name)


class AffineWarpConstraints(object):
  """Affine warp contraints class.

  `AffineWarpConstraints` allow for very succinct definitions of constraints on
  the values of entries in affine transform matrices.
  """

  def __init__(self, constraints=((None,) * 3,) * 2):
    """Creates a constraint definition for an affine transformation.

    Args:
      constraints: A doubly-nested iterable of shape `[N, N+1]` defining
        constraints on the entries of a matrix that represents an affine
        transformation in `N` dimensions. A numeric value bakes in a constraint
        on the corresponding entry in the tranformation matrix, whereas `None`
        implies that the corresponding entry will be specified at run time.

    Raises:
      TypeError: If `constraints` is not a nested iterable.
      ValueError: If the double iterable `constraints` has inconsistent
        dimensions.
    """
    try:
      self._constraints = tuple(tuple(x) for x in constraints)
    except TypeError:
      raise TypeError('constraints must be a nested iterable.')

    # Number of rows
    self._num_dim = len(self._constraints)
    expected_num_cols = self._num_dim + 1
    if any(len(x) != expected_num_cols for x in self._constraints):
      raise ValueError('The input list must define a Nx(N+1) matrix of '
                       'contraints.')

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
    """Returns the list of constraints for the i-th row of the affine matrix."""
    return self._constraints[i]

  def _combine(self, x, y):
    """Combines two constraints, raising an error if they are not compatible."""
    if x is None or y is None:
      return x or y
    if x != y:
      raise ValueError('Incompatible set of constraints provided.')
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
      new_constraints.append([self._combine(x, y) for x, y in zip(left, right)])
    return AffineWarpConstraints(new_constraints)

  # Collection of utlities to initialize an AffineGridWarper in 2D and 3D.
  @classmethod
  def no_constraints(cls, num_dim=2):
    """Empty set of constraints for a num_dim-ensional affine transform."""
    return cls(((None,) * (num_dim + 1),) * num_dim)

  @classmethod
  def translation_2d(cls, x=None, y=None):
    """Assign contraints on translation components of affine transform in 2d."""
    return cls([[None, None, x],
                [None, None, y]])

  @classmethod
  def translation_3d(cls, x=None, y=None, z=None):
    """Assign contraints on translation components of affine transform in 3d."""
    return cls([[None, None, None, x],
                [None, None, None, y],
                [None, None, None, z]])

  @classmethod
  def scale_2d(cls, x=None, y=None):
    """Assigns contraints on scaling components of affine transform in 2d."""
    return cls([[x, None, None],
                [None, y, None]])

  @classmethod
  def scale_3d(cls, x=None, y=None, z=None):
    """Assigns contraints on scaling components of affine transform in 3d."""
    return cls([[x, None, None, None],
                [None, y, None, None],
                [None, None, z, None]])

  @classmethod
  def shear_2d(cls, x=None, y=None):
    """Assigns contraints on shear components of affine transform in 2d."""
    return cls([[None, x, None],
                [y, None, None]])

  @classmethod
  def no_shear_2d(cls):
    return cls.shear_2d(x=0, y=0)

  @classmethod
  def no_shear_3d(cls):
    """Assigns contraints on shear components of affine transform in 3d."""
    return cls([[None, 0, 0, None],
                [0, None, 0, None],
                [0, 0, None, None]])
