import tensorflow as tf
import spatial_transformer_3d


def random_transform2(vol1, vol2, transform_vector, output_size):
    vol1 = spatial_transformer_3d.transformer(vol1, transform_vector, output_size, 'linear')
    vol2 = tf.to_int32(spatial_transformer_3d.transformer(vol2, transform_vector, output_size, 'nearest'))
    return vol1, vol2


def random_transform1(vol1, transform_vector, output_size):
    vol1 = spatial_transformer_3d.transformer(vol1, transform_vector, output_size, 'linear')
    return vol1


def random_transform3(vol1, vol2, vol3, transform_vector, output_size):
    vol1 = spatial_transformer_3d.transformer(vol1, transform_vector, output_size, 'linear')
    vol2 = tf.to_int32(spatial_transformer_3d.transformer(vol2, transform_vector, output_size, 'nearest'))
    vol3 = spatial_transformer_3d.transformer(vol3, transform_vector, output_size, 'linear')
    return vol1, vol2, vol3


def resize_volume(image, size, method=0, name='resizeVolume'):
    with tf.variable_scope(name):
        # size is [depth, height width]
        # image is Tensor with shape [batch, depth, height, width, channels]
        reshaped2D = tf.reshape(image, [-1, int(image.get_shape()[2]), int(image.get_shape()[3]), int(image.get_shape()[4])])
        resized2D = tf.image.resize_images(reshaped2D,[size[1],size[2]],method)
        reshaped3D = tf.reshape(resized2D, [int(image.get_shape()[0]), int(image.get_shape()[1]), size[1], size[2], int(image.get_shape()[4])])
        permuted = tf.transpose(reshaped3D, [0,3,2,1,4])
        reshaped2DB = tf.reshape(permuted, [-1, size[1], int(image.get_shape()[1]), int(image.get_shape()[4])])
        resized2DB = tf.image.resize_images(reshaped2DB,[size[1],size[0]],method)
        reshaped3DB = tf.reshape(resized2DB, [int(image.get_shape()[0]), size[2], size[1], size[0], int(image.get_shape()[4])])
        return tf.transpose(reshaped3DB, [0, 3, 2, 1, 4])


def resample_linear(inputs, sample_coords, boundary='ZERO'):

    input_size = inputs.get_shape().as_list()[1:-1]
    spatial_rank = inputs.get_shape().ndims - 2
    xy = tf.unstack(sample_coords, axis=len(sample_coords.get_shape())-1)
    index_voxel_coords = [tf.floor(x) for x in xy]

    def boundary_replicate(sample_coords0, input_size0):
        return tf.maximum(tf.minimum(sample_coords0, input_size0 - 1), 0)
    spatial_coords = [boundary_replicate(tf.cast(x, tf.int32), input_size[idx])
                      for idx, x in enumerate(index_voxel_coords)]
    spatial_coords_plus1 = [boundary_replicate(tf.cast(x+1., tf.int32), input_size[idx])
                            for idx, x in enumerate(index_voxel_coords)]

    weight = [tf.expand_dims(x - tf.cast(i, tf.float32), -1) for x, i in zip(xy, spatial_coords)]
    weight_c = [tf.expand_dims(tf.cast(i, tf.float32) - x, -1) for x, i in zip(xy, spatial_coords_plus1)]

    sz = spatial_coords[0].get_shape().as_list()
    batch_coords = tf.tile(tf.reshape(tf.range(sz[0]), [sz[0]] + [1] * (len(sz) - 1)), [1] + sz[1:])
    sc = (spatial_coords, spatial_coords_plus1)
    binary_codes = [[int(c) for c in format(i, '0%ib' % spatial_rank)] for i in range(2**spatial_rank)]

    make_sample = lambda bc: tf.gather_nd(inputs, tf.stack([batch_coords] + [sc[c][i] for i, c in enumerate(bc)], -1))
    samples = [make_sample(bc) for bc in binary_codes]

    def pyramid_combination(samples0, weight0, weight_c0):
        if len(weight0) == 1:
            return samples0[0]*weight_c0[0]+samples0[1]*weight0[0]
        else:
            return pyramid_combination(samples0[::2], weight0[:-1], weight_c0[:-1]) * weight_c0[-1] + \
                   pyramid_combination(samples0[1::2], weight0[:-1], weight_c0[:-1]) * weight0[-1]

    return pyramid_combination(samples, weight, weight_c)


def get_reference_grid(grid_size):
    return tf.to_float(tf.stack(tf.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=3))


def gradient_dx(fv):
    return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2


def gradient_dy(fv):
    return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2


def gradient_dz(fv):
    return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2


def gradient_txyz(Txyz, fname):
    return tf.stack([fname(Txyz[:, :, :, :, i]) for i in [0, 1, 2]], axis=4)


def compute_gradient_l2norm(displacement):
    dTdx = gradient_txyz(displacement, gradient_dx)
    dTdy = gradient_txyz(displacement, gradient_dy)
    dTdz = gradient_txyz(displacement, gradient_dz)
    return tf.reduce_mean(dTdx**2 + dTdy**2 + dTdz**2, [1, 2, 3, 4])


def compute_bending_energy(displacement):
    dTdx = gradient_txyz(displacement, gradient_dx)
    dTdy = gradient_txyz(displacement, gradient_dy)
    dTdz = gradient_txyz(displacement, gradient_dz)
    dTdxx = gradient_txyz(dTdx, gradient_dx)
    dTdyy = gradient_txyz(dTdy, gradient_dy)
    dTdzz = gradient_txyz(dTdz, gradient_dz)
    dTdxy = gradient_txyz(dTdx, gradient_dy)
    dTdyz = gradient_txyz(dTdy, gradient_dz)
    dTdxz = gradient_txyz(dTdx, gradient_dz)
    return tf.reduce_mean(dTdxx**2 + dTdyy**2 + dTdzz**2 + 2*dTdxy**2 + 2*dTdxz**2 + 2*dTdyz**2, [1, 2, 3, 4])


def compute_dice(input1, input2):
    mask1 = input1 >= 0.5
    mask2 = input2 >= 0.5
    vol1 = tf.reduce_sum(tf.to_float(mask1), axis=[1, 2, 3])
    vol2 = tf.reduce_sum(tf.to_float(mask2), axis=[1, 2, 3])
    dice = tf.reduce_sum(tf.to_float(mask1 & mask2), axis=[1, 2, 3])*2 / (vol1+vol2)
    return dice, vol1, vol2


def compute_centroid_distance(input1, input2, grid):

    def compute_centroid(mask, grid0):
        return tf.stack([tf.reduce_mean(tf.boolean_mask(grid0, mask[i,:,:,:,0]>=0.5), axis=0) for i in range(mask.shape[0].value)], axis=0)

    c1 = compute_centroid(input1, grid)
    c2 = compute_centroid(input2, grid)
    return tf.sqrt(tf.reduce_sum(tf.square(c1-c2), axis=1))


