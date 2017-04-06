import tensorflow as tf
from six.moves import range

from network.base_layer import BaseLayer
from network.highres3dnet import HighRes3DNet


class ScaleNet(HighRes3DNet):
    def __init__(self,
                 batch_size,
                 image_size,
                 label_size,
                 num_classes,
                 is_training=True,
                 device_str="cpu"):
        super(ScaleNet, self).__init__(batch_size,
                                           image_size,
                                           label_size,
                                           num_classes,
                                           is_training,
                                           device_str)
        self.num_res_blocks = [3, 3, 3]
        self.num_features = [16, 32, 64, 80]
        self.set_activation_type('relu')
        self.num_scale_res_block = 1
        self.merging_type = 'average'
        self.name = "ScaleNet\n" \
                    "{} scalable multiroot blocks with {} features\n" \
                    "{} dilat-0 blocks with {} features\n" \
                    "{} dilat-2 blocks with {} features\n" \
                    "{} dilat-4 blocks with {} features\n" \
                    "{} FC features to classify {} classes".format(
            self.num_scale_res_block, self.num_features[0],
            self.num_res_blocks[0], self.num_features[0],
            self.num_res_blocks[1], self.num_features[1],
            self.num_res_blocks[2], self.num_features[2],
            self.num_features[3], num_classes)
        print('using {}'.format(self.name))


    def inference(self, images, layer_id=None):
        BaseLayer._print_activations(images)
        zero_paddings = [[0, 0], [0, 0], [0, 0]]
        ########################################
        # Back End
        # images shape dimension must be [d_batch, d_z, d_y, d_x, d_mod]
        roots = tf.unstack(images, axis=4)
        nroots = len(roots)
        for r in range(nroots):
            with tf.variable_scope('mod%s_conv_1_1' % r) as scope:
                roots[r] = tf.expand_dims(roots[r], axis=4)
                roots[r] = self.conv_3x3(roots[r], 1, self.num_features[0])
                roots[r] = self.batch_norm(roots[r])
                roots[r] = self.nonlinear_acti(roots[r])
                BaseLayer._print_activations(roots[r])
        with tf.variable_scope('scalable_res') as scope:
            roots = self._scalable_multiroots_res_block(roots, nroots, self.num_features[0],
                                                        nroots, self.num_features[0],
                                                        self.num_scale_res_block)
            merged_root = self._merge_roots(roots)

        ########################################
        # Front End
        # following layers are the same as for highres3dnet
        with tf.variable_scope('res_1') as scope:
            res_1 = self._res_block(merged_root,
                                    self.num_features[0],
                                    self.num_features[0],
                                    self.num_res_blocks[0])

        ## convolutions  dilation factor = 2
        with tf.variable_scope('dilate_1_start') as scope:
            res_1 = tf.space_to_batch_nd(res_1, [2, 2, 2], zero_paddings)
            BaseLayer._print_activations(res_1)
        with tf.variable_scope('res_2') as scope:
            res_2 = self._res_block(res_1,
                                    self.num_features[0],
                                    self.num_features[1],
                                    self.num_res_blocks[1])
        with tf.variable_scope('dilate_1_end') as scope:
            res_2 = tf.batch_to_space_nd(res_2, [2, 2, 2], zero_paddings)
            BaseLayer._print_activations(res_2)

        ## convolutions  dilation factor = 4
        with tf.variable_scope('dilate_2_start') as scope:
            res_2 = tf.space_to_batch_nd(res_2, [4, 4, 4], zero_paddings)
            BaseLayer._print_activations(res_2)
        with tf.variable_scope('res_3') as scope:
            res_3 = self._res_block(res_2,
                                    self.num_features[1],
                                    self.num_features[2],
                                    self.num_res_blocks[2])
        with tf.variable_scope('dilate_2_end') as scope:
            res_3 = tf.batch_to_space_nd(res_3, [4, 4, 4], zero_paddings)
            BaseLayer._print_activations(res_3)

        ## 1x1x1 convolution "fully connected"
        with tf.variable_scope('conv_fc_1') as scope:
            conv_fc = self.conv_layer_1x1(res_3,
                                          self.num_features[2],
                                          self.num_features[3],
                                          bn=True, acti=True)
            BaseLayer._print_activations(conv_fc)

        with tf.variable_scope('conv_fc_2') as scope:
            conv_fc = self.conv_layer_1x1(conv_fc,
                                          self.num_features[3],
                                          self.num_classes,
                                          bn=True, acti=False)
            BaseLayer._print_activations(conv_fc)

        if layer_id == 'conv_features':
            return res_3

        if layer_id is None:
            return conv_fc


    def _scalable_multiroots_res_block(self, roots, nroots_in, nfea_in,
                                       nroots_out, nfea_out, n_layers):
        if n_layers == 0:
            return roots
        for layer in range(n_layers):
            # Permute root dimension and feature dimension
            fea_roots = tf.unstack(tf.stack(roots, axis=5), axis=4)
            # Cross roots res block for each feature in fea_roots
            for fea in range(nfea_in):
                with tf.variable_scope('fea_root%s_%s' % (fea, layer)):
                    fea_roots[fea] = self._res_block(fea_roots[fea], nroots_in, nroots_out,
                                                     n_blocks=1, conv_type=("3x3", "1x1"))
            nroots_in = nroots_out
            # Permute root dimension and feature dimension
            roots = tf.unstack(tf.stack(fea_roots, axis=5), axis=4)
            # Cross features res block for each root in roots
            for r in range(nroots_in):
                with tf.variable_scope('root%s_%s' % (r, layer)):
                    roots[r] = self._res_block(roots[r], nfea_in, nfea_out,
                                               n_blocks=1, conv_type=("3x3", "1x1"))
            nfea_in = nfea_out
        return roots


    def _merge_roots(self, roots):
        if self.merging_type == 'maxout':
            merged_out = tf.reduce_max(tf.stack(roots, axis=-1), axis=-1)
        else: # default is 'average'
            merged_out = tf.reduce_mean(tf.stack(roots, axis=-1), axis=-1)
        return merged_out
