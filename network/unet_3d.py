import tensorflow as tf
from base_layer import BaseLayer
from network.net_template import NetTemplate

class U_Net_3D(NetTemplate):
    def __init__(self,
                 batch_size,
                 image_size,
                 label_size,
                 num_classes,
                 is_training=True,
                 device_str="cpu"):
        super(U_Net_3D, self).__init__(batch_size,
                                       image_size,
                                       label_size,
                                       num_classes,
                                       is_training,
                                       device_str)
        assert(image_size % 8 == 0) # image_size should be divisible by 8
        self.num_fea = [32, 64, 128, 256, 512]
        self.set_activation_type('relu')
        self.name = "3D U-net"
        print "{}\n"\
              "{} kernels\n" \
              "Classifiying {} classes".format(
                  self.name, self.num_fea, self.num_classes)


    def inference(self, images, layer_id=None):
        BaseLayer._print_activations(images)
        print ""
        images = tf.expand_dims(images, 4)

        # Left - two convolutions - first level
        with tf.variable_scope('L1') as scope:
            conv_1 = self._conv_bn(images, 1, self.num_fea[1], self.num_fea[0])
            pool_1 = self.downsample_2x2(conv_1)

        # Left - two convolutions - second level
        with tf.variable_scope('L2') as scope:
            conv_2 = self._conv_bn(pool_1, self.num_fea[1], self.num_fea[2])
            pool_2 = self.downsample_2x2(conv_2)

        # Left - two convolutions - third level
        with tf.variable_scope('L3') as scope:
            conv_3 = self._conv_bn(pool_2, self.num_fea[2], self.num_fea[3])
            pool_3 = self.downsample_2x2(conv_3)

        # Left - two convolutions - fourth level
        with tf.variable_scope('L4') as scope:
            conv_4 = self._conv_bn(pool_3, self.num_fea[3], self.num_fea[4])
            up_3 = self._upsample_bn(conv_4)

        # Right - two convolutions - third level
        with tf.variable_scope('R3') as scope:
            conv_5 = self._conv_bn(tf.concat([conv_3, up_3], 4),
                                   self.num_fea[3] + self.num_fea[4],
                                   self.num_fea[3])
            up_2 = self._upsample_bn(conv_5)

        # Right - two convolutions - second level
        with tf.variable_scope('R2') as scope:
            conv_6 = self._conv_bn(tf.concat([conv_2, up_2], 4),
                                   self.num_fea[2] + self.num_fea[3],
                                   self.num_fea[2])
            up_1 = self._upsample_bn(conv_6)

        # Right - two convolutions - first level
        with tf.variable_scope('R1') as scope:
            conv_7 = self._conv_bn(tf.concat([conv_1, up_1], 4),
                                   self.num_fea[1] + self.num_fea[2],
                                   self.num_fea[1])

        # final 1x1x1 convolutions
        with tf.variable_scope('FC') as scope:
            conv_fc = self.conv_layer_1x1(conv_7,
                                          self.num_fea[1],
                                          self.num_classes,
                                          bn=True, acti=False)
        BaseLayer._print_activations(conv_fc)
        print ""

        if layer_id is None:
            return conv_fc

    def _conv_bn(self, f_in, ni_, no_, n_middle=None):
        if n_middle is None:
            n_middle = no_ if ni_ > no_ else ni_
        with tf.variable_scope('a') as scope:
            f_in = self.conv_3x3(f_in, ni_, n_middle)
            f_in = self.batch_norm(f_in)
        with tf.variable_scope('b') as scope:
            f_in = self.conv_3x3(f_in, n_middle, no_)
            f_in = self.batch_norm(f_in)
        BaseLayer._print_activations(f_in)
        print ""
        return f_in

    def _upsample_bn(self, i_):
        up_conv = self.upsample_2x2(i_)
        up_conv = self.batch_norm(up_conv)
        return up_conv
