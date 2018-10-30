# -*- coding: utf-8 -*-
# Copyright 2018 The Sonnet Authors. All Rights Reserved.
# Modifications copyright 2018 The NiftyNet Authors.
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

import numpy as np
import tensorflow as tf

from niftynet.layer.grid_warper import GridWarperLayer
from niftynet.layer.resampler import ResamplerLayer

SUPPORTED_INTERPOLATION = set(['BSPLINE', 'LINEAR', 'NEAREST'])
SUPPORTED_BOUNDARY = set(['ZERO', 'REPLICATE', 'CIRCULAR', 'SYMMETRIC'])


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
      interpolation: type_str of interpolation as used by tf.image.resize_images
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
    batch_size=int(field.shape.as_list()[0])
    spatial_rank = int(field.shape.as_list()[-1])
    resampled_list=[tf.nn.conv3d(field[:, :, :, :, d:d + 1], self._psi, strides=[1]*5, padding='VALID')
                    for d in [0, 1, 2]]
    resampled=tf.stack(resampled_list,5)
    permuted_shape=[batch_size]+[f-3 for f in self._coeff_shape]+self._knot_spacing+[spatial_rank]
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
      interpolation: type_str of interpolation as used by tf.image.resize_images
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
                                           name=name)
  def layer_op(self,field):
    input_shape = tf.shape(field)
    input_dtype = field.dtype.as_numpy_dtype
    batch_size = int(field.shape[0])
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
      interpolation: type_str of interpolation as used by tf.image.resize_images
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
    batch_size = int(field.shape[0])

    # transform grid into field coordinate space if necessary
    if self._field_transform==None:
      coords=self._psi
    else:
      coords = tf.matmul(self._psi,self._field_transform[:,:,1:3])
    # resample
    coords = tf.reshape(tf.tile(coords,[batch_size,1,1]),[-1]+list(self._output_shape)+[len(self._source_shape)])
    resampled_coords = self._resampler(field, coords)
    return resampled_coords

