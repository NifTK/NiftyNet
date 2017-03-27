import tensorflow as tf
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
        self.name = "HighRes3DNet\n"\
            "{} dilat-0 blocks with {} features\n"\
            "{} dilat-2 blocks with {} features\n"\
            "{} dilat-4 blocks with {} features\n"\
            "{} FC features to classify {} classes".format(
                self.num_res_blocks[0], self.num_features[0],
                self.num_res_blocks[1], self.num_features[1],
                self.num_res_blocks[2], self.num_features[2],
                self.num_features[3], num_classes)
        print 'using {}'.format(self.name)

    def inference(self, images, layer_id=None):
        BaseLayer._print_activations(images)
        print ""
        zero_paddings = [[0, 0], [0, 0], [0, 0]]
        images = tf.expand_dims(images, 4)
        with tf.variable_scope('conv_1_1') as scope:
            conv_1_1 = self.conv_3x3(images, 1, self.num_features[0])
            conv_1_1 = self.batch_norm(conv_1_1)
            conv_1_1 = self.nonlinear_acti(conv_1_1)
            BaseLayer._print_activations(conv_1_1)
            print ""

        with tf.variable_scope('res_1') as scope:
            res_1 = self._res_block_2_layers(conv_1_1,
                                             self.num_features[0],
                                             self.num_features[0],
                                             self.num_res_blocks[0])

        ## convolutions  dilation factor = 2
        with tf.variable_scope('dilate_1_start') as scope:
            res_1 = tf.space_to_batch_nd(res_1, [2, 2, 2], zero_paddings)
            BaseLayer._print_activations(res_1)
            print ""
        with tf.variable_scope('res_2') as scope:
            res_2 = self._res_block_2_layers(res_1,
                                             self.num_features[0],
                                             self.num_features[1],
                                             self.num_res_blocks[1])
        with tf.variable_scope('dilate_1_end') as scope:
            res_2 = tf.batch_to_space_nd(res_2, [2, 2, 2], zero_paddings)
            BaseLayer._print_activations(res_2)
            print ""

        ## convolutions  dilation factor = 4
        with tf.variable_scope('dilate_2_start') as scope:
            res_2 = tf.space_to_batch_nd(res_2, [4, 4, 4], zero_paddings)
            BaseLayer._print_activations(res_2)
            print ""
        with tf.variable_scope('res_3') as scope:
            res_3 = self._res_block_2_layers(res_2,
                                             self.num_features[1],
                                             self.num_features[2],
                                             self.num_res_blocks[2])
        with tf.variable_scope('dilate_2_end') as scope:
            res_3 = tf.batch_to_space_nd(res_3, [4, 4, 4], zero_paddings)
            BaseLayer._print_activations(res_3)
            print ""

        ## 1x1x1 convolution "fully connected"
        with tf.variable_scope('conv_fc_1') as scope:
            conv_fc = self.conv_layer_1x1(res_3,
                                          self.num_features[2],
                                          self.num_features[3],
                                          bn=True, acti=True)
            BaseLayer._print_activations(conv_fc)
            print ""

        with tf.variable_scope('conv_fc_2') as scope:
            conv_fc = self.conv_layer_1x1(conv_fc,
                                          self.num_features[3],
                                          self.num_classes,
                                          bn=True, acti=False)
            BaseLayer._print_activations(conv_fc)
            print ""

        if layer_id == 'conv_features':
            return res_3

        if layer_id is None:
            return conv_fc


    def _res_block_2_layers(self, f_in, ni_, no_, n_layers):
        if n_layers == 0:
            return f_in
        for layer in xrange(0, n_layers):
            with tf.variable_scope('block_a_%d'%layer) as scope:
                f_out = self.batch_norm(f_in)
                f_out = self.nonlinear_acti(f_out)
                f_out = self.conv_3x3(f_out, ni_, no_)
            with tf.variable_scope('block_b_%d'%layer) as scope:
                f_out = self.batch_norm(f_out)
                f_out = self.nonlinear_acti(f_out)
                f_out = self.conv_3x3(f_out, no_, no_)
            with tf.variable_scope('shortcut_%d'%layer) as scope:
                if ni_ == no_:
                    f_in = f_out + f_in
                elif ni_ < no_: # pad 0s in the feature channel dimension
                    pad_1 = (no_ - ni_)//2
                    pad_2 = no_ - ni_ - pad_1
                    shortcut = tf.pad(
                        f_in, [[0, 0], [0, 0], [0, 0], [0, 0], [pad_1, pad_2]])
                    f_in = f_out + shortcut
                elif ni_ > no_: # make a projection
                    shortcut = self.conv_1x1(f_in, ni_, no_)
                    f_in = f_out + shortcut
            ni_ = no_
        BaseLayer._print_activations(f_in)
        print '//repeated {:d} times'.format(n_layers*2)
        return f_in
