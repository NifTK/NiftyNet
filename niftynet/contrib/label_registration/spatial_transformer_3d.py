# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf


def transformer(U, theta, out_size, interpolation_method='linear', name='SpatialTransformer', **kwargs):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.
    Further edited by Eli Gibson for 3D transforms.
    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, depth, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 12].
    out_size: tuple of two ints
        The size of the output of the network (height, width, depth)
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py
    .. [3]  https://github.com/daviddao/spatial-transformer-tensorflow/blob/master/spatial_transformer.py
    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0., 0.],
                             [0., 1., 0., 0.],
                             [0., 0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)
    """

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, z, out_size, method):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            depth = tf.shape(im)[1]
            height = tf.shape(im)[2]
            width = tf.shape(im)[3]
            channels = im.get_shape()[4]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            z = tf.cast(z, 'float32')
            depth_f = tf.cast(depth, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_depth = out_size[0]
            out_height = out_size[1]
            out_width = out_size[2]
            zero = tf.zeros([], dtype='int32')
            max_z = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_y = tf.cast(tf.shape(im)[2] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[3] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0
            z = (z + 1.0)*(depth_f) / 2.0

            # do sampling
            if method=='linear':
                x0 = tf.cast(tf.floor(x), 'int32')
                x1 = x0 + 1
                y0 = tf.cast(tf.floor(y), 'int32')
                y1 = y0 + 1
                z0 = tf.cast(tf.floor(z), 'int32')
                z1 = z0 + 1

                x0 = tf.clip_by_value(x0, zero, max_x)
                x1 = tf.clip_by_value(x1, zero, max_x)
                y0 = tf.clip_by_value(y0, zero, max_y)
                y1 = tf.clip_by_value(y1, zero, max_y)
                z0 = tf.clip_by_value(z0, zero, max_z)
                z1 = tf.clip_by_value(z1, zero, max_z)
                dim3 = width
                dim2 = width*height
                dim1 = width*height*depth
                base = _repeat(tf.range(num_batch)*dim1, out_depth*out_height*out_width)
                base_z0 = base + z0*dim2
                base_z1 = base + z1*dim2
                base_z0y0 = base_z0 + y0*dim3
                base_z0y1 = base_z0 + y1*dim3
                base_z1y0 = base_z1 + y0*dim3
                base_z1y1 = base_z1 + y1*dim3
                idx_a = base_z0y0 + x0
                idx_b = base_z0y1 + x0
                idx_c = base_z0y0 + x1
                idx_d = base_z0y1 + x1
                idx_e = base_z1y0 + x0
                idx_f = base_z1y1 + x0
                idx_g = base_z1y0 + x1
                idx_h = base_z1y1 + x1

                # use indices to lookup pixels in the flat image and restore
                # channels dim
                im_flat = tf.reshape(im, [-1, int(channels)])
                im_flat = tf.cast(im_flat, 'float32')
                Ia = tf.gather(im_flat, idx_a)
                Ib = tf.gather(im_flat, idx_b)
                Ic = tf.gather(im_flat, idx_c)
                Id = tf.gather(im_flat, idx_d)
                Ie = tf.gather(im_flat, idx_e)
                If = tf.gather(im_flat, idx_f)
                Ig = tf.gather(im_flat, idx_g)
                Ih = tf.gather(im_flat, idx_h)

                # and finally calculate interpolated values
                x0_f = tf.cast(x0, 'float32')
                x1_f = x0_f+1.  # Note that we use x0f+1 instead of cast(x1) to avoid problems with sampling from 1 pixel wide images.
                y0_f = tf.cast(y0, 'float32')
                y1_f = y0_f+1.
                z0_f = tf.cast(z0, 'float32')
                z1_f = z0_f+1.
                wa = tf.expand_dims(((x1_f-x) * (y1_f-y) * (z1_f-z)), 1)
                wb = tf.expand_dims(((x1_f-x) * (y-y0_f) * (z1_f-z)), 1)
                wc = tf.expand_dims(((x-x0_f) * (y1_f-y) * (z1_f-z)), 1)
                wd = tf.expand_dims(((x-x0_f) * (y-y0_f) * (z1_f-z)), 1)
                we = tf.expand_dims(((x1_f-x) * (y1_f-y) * (z-z0_f)), 1)
                wf = tf.expand_dims(((x1_f-x) * (y-y0_f) * (z-z0_f)), 1)
                wg = tf.expand_dims(((x-x0_f) * (y1_f-y) * (z-z0_f)), 1)
                wh = tf.expand_dims(((x-x0_f) * (y-y0_f) * (z-z0_f)), 1)
                output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id, we*Ie, wf*If, wg*Ig, wh*Ih])
            elif method=='nearest':
                x0 = tf.cast(tf.round(x), 'int32')
                y0 = tf.cast(tf.round(y), 'int32')
                z0 = tf.cast(tf.round(z), 'int32')
                x0 = tf.clip_by_value(x0, zero, max_x)
                y0 = tf.clip_by_value(y0, zero, max_y)
                z0 = tf.clip_by_value(z0, zero, max_z)
                dim3 = width
                dim2 = width*height
                dim1 = width*height*depth
                base = _repeat(tf.range(num_batch)*dim1, out_depth*out_height*out_width)
                idx_a = base + z0*dim2 + y0*dim3 + x0
                im_flat = tf.reshape(im, [-1, int(channels)])
                output = tf.gather(im_flat, idx_a)
            else:
                raise ValueError('Unknown interpolation type')
            return output

    def _meshgrid(depth, height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t, z_t = np.meshgrid(np.linspace(-1, 1, width),
            #                              np.linspace(-1, 1, height),
            #                              np.linspace(-1, 1, depth))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), z_t.flatten(), ones])
            if height==1:
                y_t = tf.tile(tf.expand_dims(tf.expand_dims(tf.zeros(1),0),2),tf.stack([depth,1,width]))
            else:
                y_t = tf.tile(tf.expand_dims(tf.expand_dims(tf.linspace(-1.0, 1.0, height),0),2),tf.stack([depth,1,width]))
            if depth==1:
                z_t = tf.tile(tf.expand_dims(tf.expand_dims(tf.zeros(1),1),2),tf.stack([1,height,width]))
            else:
                z_t = tf.tile(tf.expand_dims(tf.expand_dims(tf.linspace(-1.0, 1.0, depth),1),2),tf.stack([1,height,width]))
            if width==1:
                x_t = tf.tile(tf.expand_dims(tf.expand_dims(tf.zeros(1),0),1),tf.stack([depth,height,1]))
            else:
                x_t = tf.tile(tf.expand_dims(tf.expand_dims(tf.linspace(-1.0, 1.0, width),0),1),tf.stack([depth,height,1]))
            
            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            z_t_flat = tf.reshape(z_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat([x_t_flat, y_t_flat, z_t_flat, ones], axis=0)
            return grid

    def _transform(theta, input_dim, out_size, method):
        with tf.variable_scope('_transform'):
            num_batch = input_dim.get_shape()[0]  #previously: num_batch = tf.shape(input_dim)[0]
            depth = input_dim.get_shape()[1]
            height = input_dim.get_shape()[2]
            width = input_dim.get_shape()[3]
            num_channels = input_dim.get_shape()[4]
            theta = tf.reshape(theta, (-1, 3, 4))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, z_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            depth_f = tf.cast(depth, 'float32')
            out_depth = out_size[0]
            out_height = out_size[1]
            out_width = out_size[2]
            grid = _meshgrid(out_depth, out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, [int(num_batch)])  #previously grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, [int(num_batch), 4, -1])
            
            # Transform A x (x_t, y_t, z_t, 1)^T -> (x_s, y_s, z_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            z_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            z_s_flat = tf.reshape(z_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                input_dim, x_s_flat, y_s_flat, z_s_flat, 
                out_size,method)
            output = tf.reshape(
                input_transformed, tf.stack([int(num_batch), int(out_depth), int(out_height), int(out_width), int(num_channels)]))
            return output

    with tf.variable_scope(name):
        output = _transform(theta, U, out_size, interpolation_method)
        return output


def batch_transformer(U, thetas, out_size, interpolation_method='linear', name='BatchSpatialTransformer'):
    """Batch Spatial Transformer Layer
    Parameters
    ----------
    U : float
        tensor of inputs [num_batch,height,width,depth,num_channels]
    thetas : float
        a set of transformations for each input [num_batch,num_transforms,12]
    out_size : int
        the size of the output [out_height,out_width,out_depth]
    Returns: float
        Tensor of size [num_batch*num_transforms,out_height,out_width,out_depth,num_channels]
    """
    with tf.variable_scope(name):
        num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i]*num_transforms for i in range(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        return transformer(input_repeated, thetas, out_size, interpolation_method)
