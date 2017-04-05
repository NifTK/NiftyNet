import tensorflow as tf
from six.moves import range

from base_layer import BaseLayer
from network.net_template import NetTemplate


class HighRes3DNet(NetTemplate):
    def __init__(self,
                 batch_size,
                 image_size,
                 label_size,
                 num_classes,
                 is_training=True,
                 device_str="cpu"):
        super(HighRes3DNet, self).__init__(batch_size,
                                           image_size,
                                           label_size,
                                           num_classes,
                                           is_training,
                                           device_str)
        self.num_res_blocks = [3, 3, 3]
        self.num_features = [16, 32, 64, 80]
        self.set_activation_type('relu')
        self.name = "HighRes3DNet\n" \
                    "{} dilat-0 blocks with {} features\n" \
                    "{} dilat-2 blocks with {} features\n" \
                    "{} dilat-4 blocks with {} features\n" \
                    "{} FC features to classify {} classes".format(
            self.num_res_blocks[0], self.num_features[0],
            self.num_res_blocks[1], self.num_features[1],
            self.num_res_blocks[2], self.num_features[2],
            self.num_features[3], num_classes)
        print('using {}'.format(self.name))

    def inference(self, images, layer_id=None):
        BaseLayer._print_activations(images)
        print("")
        zero_paddings = [[0, 0], [0, 0], [0, 0]]
        with tf.variable_scope('conv_1_1') as scope:
            conv_1_1 = self.conv_3x3(images, 1, self.num_features[0])
            conv_1_1 = self.batch_norm(conv_1_1)
            conv_1_1 = self.nonlinear_acti(conv_1_1)
            BaseLayer._print_activations(conv_1_1)
            print("")

        with tf.variable_scope('res_1') as scope:
            res_1 = self._res_block(conv_1_1,
                                    self.num_features[0],
                                    self.num_features[0],
                                    self.num_res_blocks[0])

        ## convolutions  dilation factor = 2
        with tf.variable_scope('dilate_1_start') as scope:
            res_1 = tf.space_to_batch_nd(res_1, [2, 2, 2], zero_paddings)
            BaseLayer._print_activations(res_1)
            print("")
        with tf.variable_scope('res_2') as scope:
            res_2 = self._res_block(res_1,
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
            res_3 = self._res_block(res_2,
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

    def _res_block(self, f_in, ni_, no_, n_blocks, conv_type=("conv_3x3", "conv_3x3")):
        n_layers = len(conv_type)
        conv = []
        for l in range(n_layers):
            if conv_type[l] == "conv_3x3":
                conv_layer_l = self.conv_3x3
            elif conv_type[l] == "conv_1x1":
                conv_layer_l = self.conv_1x1
            else:
                raise ValueError('Convolution type %s not supported' % conv_type[l])
            conv.append(conv_layer_l)
        if n_blocks == 0:
            return f_in
        for block in range(0, n_blocks):
            # f_out as to be reinitialized to f_in at the beginning of each block
            # because of the shortcut connection
            f_out = f_in
            for layer in range(n_layers):
                with tf.variable_scope('block_%s_%d' % (chr(ord('a')+layer), block)) as scope:
                    f_out = self.batch_norm(f_out)
                    f_out = self.nonlinear_acti(f_out)
                    if layer == 0:
                        f_out = conv[layer](f_out, ni_, no_)
                    else:
                        f_out = conv[layer](f_out, no_, no_)
            with tf.variable_scope('shortcut_%d' % block) as scope:
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
        print('//repeated {:d} times'.format(n_blocks * n_layers))
        return f_in
