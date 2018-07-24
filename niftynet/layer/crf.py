# -*- coding: utf-8 -*-
"""
Re-implementation of [1] in Tensorflow for volumetric image processing.

[1] Zheng et al.
"Conditional random fields as recurrent neural networks." ICCV 2015.
https://arxiv.org/abs/1502.03240
"""
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.layer_util import infer_spatial_rank, expand_spatial_params


class CRFAsRNNLayer(TrainableLayer):
    """
    This class defines a layer implementing CRFAsRNN described in [1] using
    a bilateral and a spatial kernel as in [2].
    Essentially, this layer smooths its input based on a distance in a feature
    space comprising spatial and feature dimensions.
    High-dimensional Gaussian filtering adapted from [3].

    [1] Zheng et al., https://arxiv.org/abs/1502.03240
    [2] Krahenbuhl and Koltun, https://arxiv.org/pdf/1210.5644.pdf
    [3] Adam et al., https://graphics.stanford.edu/papers/permutohedral/
    """

    def __init__(self,
                 alpha=5.,
                 beta=5.,
                 gamma=5.,
                 T=5,
                 aspect_ratio=None,
                 mu_init=None,
                 w_init=None,
                 name="crf_as_rnn"):
        """
        Currently this layer supports spatial ND dense CRF with CPU only.
        To place the layer on CPU::

            with tf.device('/cpu:0'):
                crf_layer = CRFAsRNNLayer()
                crf_output = crf_layer(features, raw_logits)

        To ensure backpropagations during training are placed on CPU as well,
        the optimiser should be used with argument
        ``colocate_gradients_with_ops=True``, e.g.,::

            train_op = tf.train.GradientDescentOptimizer(.5).minimise(
                training_loss, colocate_gradients_with_ops=True)



        :param alpha: bandwidth for spatial coordinates in bilateral kernel.
                      Higher values cause more spatial blurring
        :param beta: bandwidth for feature coordinates in bilateral kernel
                      Higher values cause more feature blurring
        :param gamma: bandwidth for spatial coordinates in spatial kernel
                      Higher values cause more spatial blurring
        :param T: number of stacked layers in the RNN
        :param aspect_ratio: spacing of adjacent voxels
            (allows isotropic spatial smoothing when voxels are not isotropic)
        :param mu_init: initial compatibility matrix [n_classes x n_classes]
        :param w_init: initial kernel weights [2 x n_classes]
            where w_init[0] are the weights for the bilateral kernel,
                  w_init[1] are the weights for the spatial kernel.
        :param name:
        """

        super(CRFAsRNNLayer, self).__init__(name=name)
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma
        self._T = T
        self._aspect_ratio = aspect_ratio
        self._mu_init = mu_init
        self._w_init = w_init

        assert self._alpha > 0, 'alpha should be positive'
        assert self._beta > 0, 'beta should be positive'
        assert self._gamma > 0, 'gamma should be positive'

    def layer_op(self, I, U):
        """
        Compute `T` iterations of mean field update given a dense CRF.

        This layer maintains trainable CRF model parameters
        (a compatibility function and `m` kernel weights).

        :param I: feature maps used in the dense pairwise term of CRF
        :param U: activation maps used in the unary term of CRF (before softmax)
        :return: Maximum a posteriori labeling (before softmax)
        """

        spatial_dim = infer_spatial_rank(U)
        all_shape = U.shape.as_list()
        batch_size, spatial_shape, n_ch = \
            all_shape[0], all_shape[1:-1], all_shape[-1]
        n_feat = I.shape.as_list()[-1]
        if self._aspect_ratio is None:
            self._aspect_ratio = [1.] * spatial_dim
        self._aspect_ratio = expand_spatial_params(
            self._aspect_ratio, spatial_dim, float)

        # constructing the scaled regular grid
        spatial_coords = tf.meshgrid(
            *[np.arange(i, dtype=np.float32) * a
                for i, a in zip(spatial_shape, self._aspect_ratio)],
            indexing='ij')
        spatial_coords = tf.stack(spatial_coords, spatial_dim)
        spatial_coords = tf.tile(
            tf.expand_dims(spatial_coords, 0),
            [batch_size] + [1] * spatial_dim + [1])
        # print(spatial_coords.shape, I.shape)

        # concatenating spatial coordinates and features
        # (and squeeze spatially)
        # for the bilateral kernel
        bilateral_coords = tf.reshape(
            tf.concat([spatial_coords / self._alpha, I / self._beta], -1),
            [batch_size, -1, n_feat + spatial_dim])
        # for the spatial kernel
        spatial_coords = tf.reshape(
            spatial_coords / self._gamma, [batch_size, -1, spatial_dim])

        # Build permutohedral structures for smoothing
        permutohedrals = [
            permutohedral_prepare(coords)
            for coords in (bilateral_coords, spatial_coords)]

        # trainable compatibility matrix mu (initialised as identity * -1)
        mu_shape = [n_ch, n_ch]
        if self._mu_init is None:
            self._mu_init = -np.eye(n_ch)
        self._mu_init = np.reshape(self._mu_init, mu_shape)
        mu = tf.get_variable(
            'Compatibility',
            initializer=tf.constant(self._mu_init, dtype=tf.float32))

        # trainable kernel weights
        weight_shape = [1] * spatial_dim + [1, n_ch]
        if self._w_init is None:
            self._w_init = [np.ones(n_ch), np.ones(n_ch)]
        for _w in self._w_init:
            # optimising (w - 1.0) so that we get rid of the
            # -Q_i term (in equ.(5) of Krahenbuhl and Koltun 2012).
            _w = np.reshape(_w, weight_shape) - 1.0
        kernel_weights = [tf.get_variable(
            'FilterWeights{}'.format(idx),
            initializer=tf.constant(self._w_init[idx], dtype=tf.float32))
            for idx, k in enumerate(permutohedrals)]

        H1 = U
        for t in range(self._T):
            H1 = ftheta(U, H1, permutohedrals, mu, kernel_weights,
                        name='{}{}'.format(self.name, t))
        return H1


def ftheta(U, H1, permutohedrals, mu, kernel_weights, name):
    """
    A mean-field update

    :param U: the unary potentials (before softmax)
    :param H1: the previous mean-field approximation to be updated
    :param permutohedrals: fixed position vectors for fast filtering
    :param mu: compatibility function
    :param kernel_weights: weights bilateral/spatial kernels
    :param name: layer name
    :return: updated mean-field distribution
    """
    batch_size, n_ch = U.shape.as_list()[0], U.shape.as_list()[-1]
    n_voxels = np.prod(U.shape.as_list()[:-1])

    H1 = tf.reshape(tf.nn.softmax(H1), [batch_size, -1, n_ch])
    Q1 = 0
    for idx, permutohedral in enumerate(permutohedrals):
        # Message Passing
        Q = _permutohedral_gen(permutohedral, H1, name + str(idx))
        Q.set_shape([n_voxels, n_ch])
        # Weighting Filtered Outputs
        Q1 = Q1 + Q * kernel_weights[idx]

    # Compatibility Transform, Adding Unary Potentials
    # output logits, not the softmax
    return U - tf.reshape(tf.matmul(Q1, mu), U.shape.as_list())


def permutohedral_prepare(position_vectors):
    """
    Embedding the position vectors in a high-dimensional space,
    the lattice points are stored in hash tables.

    The function computes:
    - translation by the nearest reminder-0
    - ranking permutation to the canonical simplex
    - barycentric weights in the canonical simplex

    :param position_vectors: N x d position
    :return: barycentric weights, blur neighbours points in the hyperplane
    """
    pos_shape = position_vectors.shape.as_list()
    batch_size, n_voxels, n_ch = pos_shape[0], pos_shape[1:-1], pos_shape[-1]
    n_voxels = np.prod(n_voxels)
    n_ch_1 = n_ch + 1

    # reshaping batches and voxels into one dimension
    # means we can use 1D gather and hashing easily
    position_vectors = tf.reshape(position_vectors, [-1, n_ch])

    # Generate position vectors in lattice space
    # first rotate position into the (n_ch+1)-dimensional hyperplane
    inv_std_dev = np.sqrt(2 / 3.) * n_ch_1
    scale_factor = [
        inv_std_dev / np.sqrt((i + 1) * (i + 2)) for i in range(n_ch)]
    Ex = [None] * n_ch_1
    Ex[n_ch] = -n_ch * position_vectors[:, n_ch - 1] * scale_factor[n_ch - 1]
    for dit in range(n_ch - 1, 0, -1):
        Ex[dit] = Ex[dit + 1] - \
                  dit * position_vectors[:, dit - 1] * scale_factor[dit - 1] + \
                  (dit + 2) * position_vectors[:, dit] * scale_factor[dit]
    Ex[0] = 2 * position_vectors[:, 0] * scale_factor[0] + Ex[1]
    Ex = tf.stack(Ex, -1)

    # Compute coordinates
    # Get closest remainder-0 point
    v = tf.to_int32(tf.round(Ex / float(n_ch_1)))
    rem0 = v * n_ch_1
    # (sumV != 0)  meaning off the plane
    sumV = tf.reduce_sum(v, 1, True)

    # Find the simplex we are in and store it in rank
    # (where rank describes what position coordinate i has
    # in the sorted order of the features values).
    # This can be done more efficiently
    # if necessary following the permutohedral paper.
    _, index = tf.nn.top_k(Ex - tf.to_float(rem0), n_ch_1, sorted=True)
    _, rank = tf.nn.top_k(-index, n_ch_1, sorted=True)

    # if the point doesn't lie on the plane (sum != 0) bring it back
    rank = rank + sumV
    add_minus_sub = \
        tf.to_int32(rank < 0) * n_ch_1 - \
        tf.to_int32(rank >= n_ch_1) * n_ch_1
    rem0 = rem0 + add_minus_sub
    rank = rank + add_minus_sub

    # Compute the barycentric coordinates (p.10 in [Adams et al 2010])
    v2 = (Ex - tf.to_float(rem0)) / float(n_ch_1)
    # CRF2RNN uses the calculated ranks to get v2 sorted in O(n_ch) time
    # We cheat here by using the easy to implement
    # but slower method of sorting again in O(n_ch log n_ch)
    # we might get this even more efficient
    # if we correct the original sorted data above
    v_sorted, _ = tf.nn.top_k(v2, n_ch_1, sorted=True)
    v_sorted = tf.reverse(v_sorted, [-1])
    # weighted against the canonical simplex vertices
    barycentric = \
        v_sorted - tf.concat([v_sorted[:, -1:] - 1., v_sorted[:, :-1]], 1)

    # Compute all vertices and their offset
    def _simple_hash(key):
        # WARNING: This hash function does not guarantee
        # uniqueness of different position_vectors
        hash_vector = np.power(
            int(np.floor(np.power(tf.int64.max, 1. / (n_ch + 2)))),
            [range(1, n_ch_1)])
        hash_vector = tf.constant(hash_vector, dtype=tf.int64)
        return tf.reduce_sum(tf.to_int64(key) * hash_vector, 1)

    hash_table = tf.contrib.lookup.MutableDenseHashTable(
        tf.int64, tf.int64,
        default_value=tf.constant([-1] * n_ch, dtype=tf.int64),
        empty_key=-1,
        initial_num_buckets=8,
        checkpoint=False)
    index_table = tf.contrib.lookup.MutableDenseHashTable(
        tf.int64, tf.int64,
        default_value=0,
        empty_key=-1,
        initial_num_buckets=8,
        checkpoint=False)

    # canonical simplex (p.4 in [Adams et al 2010])
    canonical = \
        [[i] * (n_ch_1 - i) + [i - n_ch - 1] * i for i in range(n_ch_1)]

    insert_ops = []
    loc = [None] * n_ch_1
    loc_hash = [None] * n_ch_1
    for scit in range(n_ch_1):
        # Compute the location of the lattice point explicitly
        # (all but the last coordinate -
        #  it's redundant because they sum to zero)
        loc[scit] = tf.gather(canonical[scit], rank[:, :-1]) + rem0[:, :-1]
        loc_hash[scit] = _simple_hash(loc[scit])
        insert_ops.append(
            hash_table.insert(loc_hash[scit], tf.to_int64(loc[scit])))

    with tf.control_dependencies(insert_ops):
        fused_loc_hash, fused_loc = hash_table.export()
        is_good_key = tf.where(tf.not_equal(fused_loc_hash, -1))[:, 0]
        fused_loc = tf.gather(fused_loc, is_good_key)
        fused_loc_hash = tf.gather(fused_loc_hash, is_good_key)

    # The additional index hash table is used to
    # linearise the hash table so that we can `tf.scatter` and `tf.gather`
    # (range_id 0 reserved for the indextable's default value)
    range_id = tf.range(
        1, tf.size(fused_loc_hash, out_type=tf.int64) + 1, dtype=tf.int64)
    range_id = tf.expand_dims(range_id, 1)
    insert_indices = index_table.insert(fused_loc_hash, range_id)

    # linearised [batch, spatial_dim] indices
    # where in the splat variable each simplex vertex is
    batch_index = tf.range(batch_size, dtype=tf.int32)
    batch_index = tf.expand_dims(batch_index, 0)
    batch_index = tf.tile(batch_index, [n_voxels, 1])
    batch_index = tf.to_int64(tf.reshape(batch_index, [-1]))

    indices = [None] * n_ch_1
    blur_neighbours1 = [None] * n_ch_1
    blur_neighbours2 = [None] * n_ch_1
    with tf.control_dependencies([insert_indices]):
        for dit in range(n_ch_1):
            # the neighbors along each axis.
            offset = [n_ch if i == dit else -1 for i in range(n_ch)]
            offset = tf.constant(offset, dtype=tf.int64)
            blur_neighbours1[dit] = \
                index_table.lookup(_simple_hash(fused_loc + offset))
            blur_neighbours2[dit] = \
                index_table.lookup(_simple_hash(fused_loc - offset))
            indices[dit] = tf.stack([
                index_table.lookup(loc_hash[dit]), batch_index], 1)

    return barycentric, blur_neighbours1, blur_neighbours2, indices


def permutohedral_compute(data_vectors,
                          barycentric,
                          blur_neighbours1,
                          blur_neighbours2,
                          indices,
                          name,
                          reverse):
    """
    Splat, Gaussian blur, and slice

    :param data_vectors: value map to be filtered
    :param barycentric: embedding coordinates
    :param blur_neighbours1: first neighbours' coordinates relative to indices
    :param blur_neighbours2: second neighbours' coordinates relative to indices
    :param indices: corresponding locations of data_vectors
    :param name: layer name
    :param reverse: transpose the Gaussian kernel if True
    :return: filtered data_vectors (sliced to the original space)
    """

    num_simplex_corners = barycentric.shape.as_list()[-1]
    n_ch = num_simplex_corners - 1
    batch_size = data_vectors.shape.as_list()[0]
    n_ch_data = data_vectors.shape.as_list()[-1]
    data_vectors = tf.reshape(data_vectors, [-1, n_ch_data])
    # Convert to homogeneous coordinates
    data_vectors = tf.concat(
        [data_vectors, tf.ones_like(data_vectors[:, 0:1])], 1)

    # Splatting
    with tf.variable_scope(name):
        splat = tf.contrib.framework.local_variable(
            tf.constant(0.0), validate_shape=False, name='splatbuffer')

    #with tf.control_dependencies([splat.initialized_value()]):
    initial_splat = tf.zeros(
        [tf.shape(blur_neighbours1[0])[0] + 1, batch_size, n_ch_data + 1])
    reset_splat = tf.assign(splat, initial_splat, validate_shape=False)

    with tf.control_dependencies([reset_splat]):
        for scit in range(num_simplex_corners):
            data = data_vectors * barycentric[:, scit:scit + 1]
            splat = tf.scatter_nd_add(splat, indices[scit], data)

    # Blur with 1D kernels
    for dit in range(n_ch, -1, -1) if reverse else range(n_ch + 1):
        b1 = tf.gather(splat, blur_neighbours1[dit])
        b3 = tf.gather(splat, blur_neighbours2[dit])
        splat = tf.concat([
            splat[:1, ...], splat[1:, ...] + 0.5 * (b1 + b3)], 0)

    # Alpha is a magic scaling constant from CRFAsRNN code
    alpha = 1. / (1. + np.power(2., -n_ch))
    normalized = splat[..., :-1] / (splat[..., -1:] + 1e-20)

    # Slice
    sliced = tf.gather_nd(normalized, indices[0]) * barycentric[:, :1] * alpha
    for scit in range(1, num_simplex_corners):
        sliced = sliced + \
                 tf.gather_nd(normalized, indices[scit]) * \
                 barycentric[:, scit:scit + 1] * alpha
    return sliced


def _py_func_with_grads(func, inp, Tout, stateful=True, name=None, grad=None):
    """
    To get this to work with automatic differentiation
    we use a hack attributed to Sergey Ioffe
    mentioned here: http://stackoverflow.com/questions/36456436

    Define custom _py_func_with_grads which takes also a grad op as argument:
    from https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342

    :param func:
    :param inp:
    :param Tout:
    :param stateful:
    :param name:
    :param grad:
    :return:
    """
    # Need to generate a unique name to avoid duplicates:
    import uuid
    rnd_name = 'PyFuncGrad' + str(uuid.uuid4())
    # tf.logging.info('CRFasRNN layer iteration {}'.format(rnd_name))
    tf.RegisterGradient(rnd_name)(grad)  # see _MySquareGrad for grad example
    with tf.get_default_graph().gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)[0]


def _gradient_stub(data_vectors,
                   barycentric,
                   blur_neighbours1,
                   blur_neighbours2,
                   indices,
                   name):
    """
    This is a stub operator whose purpose is
    to allow us to overwrite the gradient.
    The forward pass gives zeros and
    the backward pass gives the correct gradients
    for the permutohedral_compute function

    :param data_vectors:
    :param barycentric:
    :param blur_neighbours1:
    :param blur_neighbours2:
    :param indices:
    :param name:
    :return:
    """

    def _dummy_wrapper(data_vectors_np, *_unused):
        return np.float32(0)

    def _permutohedral_grad_wrapper(op, grad):
        # Differentiation can be done using permutohedral lattice
        # with Gaussian filter order reversed
        filtering_grad = permutohedral_compute(
            grad, op.inputs[1], op.inputs[2], op.inputs[3], op.inputs[4],
            name, reverse=True)
        return [filtering_grad] + [None for i in op.inputs[1:]]

    _inputs = [
        data_vectors, barycentric, blur_neighbours1, blur_neighbours2, indices]

    partial_grads_func = _py_func_with_grads(
        _dummy_wrapper,
        _inputs,
        [tf.float32],
        name=name,
        grad=_permutohedral_grad_wrapper)
    n_voxels = np.prod(data_vectors.shape[:-1].as_list())
    n_ch = data_vectors.shape[-1]
    partial_grads_func.set_shape([n_voxels, n_ch])
    return partial_grads_func


def _permutohedral_gen(permutohedral, data_vectors, name):
    """
    a wrapper combines permutohedral_compute and a customised gradient op.

    :param permutohedral:
    :param data_vectors:
    :param name:
    :return:
    """
    barycentric, blur_neighbours1, blur_neighbours2, indices = permutohedral
    backward_branch = _gradient_stub(
        data_vectors,
        barycentric,
        blur_neighbours1,
        blur_neighbours2,
        indices,
        name)
    forward_branch = permutohedral_compute(
        data_vectors,
        barycentric,
        blur_neighbours1,
        blur_neighbours2,
        indices,
        name,
        reverse=False)
    return backward_branch + tf.stop_gradient(forward_branch)
