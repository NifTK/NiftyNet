import tensorflow as tf
from six.moves import range

from base_layer import BaseLayer
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
        print("")
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
                print("")
        with tf.variable_scope('scalable_res') as scope:
            roots = self._scalable_multiroots_res_block(roots, nroots, self.num_features[0],
                                                        nroots, self.num_features[0],
                                                        self.num_scale_res_block)
            merged_root = self._merge_roots(roots)

        ########################################
        # Front End
        # following layers are the same as for highres3dnet
        with tf.variable_scope('res_1') as scope:
            res_1 = self._res_block_2_layers(merged_root,
                                             self.num_features[0],
                                             self.num_features[0],
                                             self.num_res_blocks[0])

        ## convolutions  dilation factor = 2
        with tf.variable_scope('dilate_1_start') as scope:
            res_1 = tf.space_to_batch_nd(res_1, [2, 2, 2], zero_paddings)
            BaseLayer._print_activations(res_1)
            print("")
        with tf.variable_scope('res_2') as scope:
            res_2 = self._res_block_2_layers(res_1,
                                             self.num_features[0],
                                             self.num_features[1],
                                             self.num_res_blocks[1])
        with tf.variable_scope('dilate_1_end') as scope:
            res_2 = tf.batch_to_space_nd(res_2, [2, 2, 2], zero_paddings)
            BaseLayer._print_activations(res_2)
            print("")

        ## convolutions  dilation factor = 4
        with tf.variable_scope('dilate_2_start') as scope:
            res_2 = tf.space_to_batch_nd(res_2, [4, 4, 4], zero_paddings)
            BaseLayer._print_activations(res_2)
            print("")
        with tf.variable_scope('res_3') as scope:
            res_3 = self._res_block_2_layers(res_2,
                                             self.num_features[1],
                                             self.num_features[2],
                                             self.num_res_blocks[2])
        with tf.variable_scope('dilate_2_end') as scope:
            res_3 = tf.batch_to_space_nd(res_3, [4, 4, 4], zero_paddings)
            BaseLayer._print_activations(res_3)
            print("")

        ## 1x1x1 convolution "fully connected"
        with tf.variable_scope('conv_fc_1') as scope:
            conv_fc = self.conv_layer_1x1(res_3,
                                          self.num_features[2],
                                          self.num_features[3],
                                          bn=True, acti=True)
            BaseLayer._print_activations(conv_fc)
            print("")

        with tf.variable_scope('conv_fc_2') as scope:
            conv_fc = self.conv_layer_1x1(conv_fc,
                                          self.num_features[3],
                                          self.num_classes,
                                          bn=True, acti=False)
            BaseLayer._print_activations(conv_fc)
            print("")

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
                    fea_roots[fea] = self._res_block_conv3_conv1(fea_roots[fea], nroots_in,
                                                                 nroots_out, n_layers=1)
            nroots_in = nroots_out
            # Permute root dimension and feature dimension
            roots = tf.unstack(tf.stack(fea_roots, axis=5), axis=4)
            # Cross features res block for each root in roots
            for r in range(nroots_in):
                with tf.variable_scope('root%s_%s' % (r, layer)):
                    roots[r] = self._res_block_conv3_conv1(roots[r], nfea_in,
                                                           nfea_out, n_layers=1)
            nfea_in = nfea_out
        return roots


    def _res_block_conv3_conv1(self, f_in, ni_, no_, n_layers):
        if n_layers == 0:
            return f_in
        for layer in range(0, n_layers):
            with tf.variable_scope('block_a_%d' % layer) as scope:
                f_out = self.batch_norm(f_in)
                f_out = self.nonlinear_acti(f_out)
                f_out = self.conv_3x3(f_out, ni_, no_)
            with tf.variable_scope('block_b_%d' % layer) as scope:
                f_out = self.batch_norm(f_out)
                f_out = self.nonlinear_acti(f_out)
                f_out = self.conv_1x1(f_out, no_, no_)
            with tf.variable_scope('shortcut_%d' % layer) as scope:
                if ni_ == no_:
                    f_in = f_out + f_in
                elif ni_ < no_:  # pad 0s in the feature channel dimension
                    pad_1 = (no_ - ni_) // 2
                    pad_2 = no_ - ni_ - pad_1
                    shortcut = tf.pad(
                        f_in, [[0, 0], [0, 0], [0, 0], [0, 0], [pad_1, pad_2]])
                    f_in = f_out + shortcut
                elif ni_ > no_:  # make a projection
                    shortcut = self.conv_1x1(f_in, ni_, no_)
                    f_in = f_out + shortcut
            ni_ = no_
        BaseLayer._print_activations(f_in)
        print('//repeated conv 3x3x3 followed by conv 1x1x1 {:d} times'.format(n_layers))
        return f_in


    def _merge_roots(self, roots):
        if self.merging_type == 'maxout':
            merged_out = tf.reduce_max(tf.stack(roots, axis=-1), axis=-1)
        else: # default is 'average'
            merged_out = tf.reduce_mean(tf.stack(roots, axis=-1), axis=-1)
        return merged_out
