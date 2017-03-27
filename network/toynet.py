import tensorflow as tf
from base_layer import BaseLayer
from network.net_template import NetTemplate

class ToyNet(NetTemplate):
    def __init__(self,
                 batch_size,
                 image_size,
                 label_size,
                 num_classes,
                 is_training=True,
                 device_str="cpu"):
        super(ToyNet, self).__init__(batch_size,
                                     image_size,
                                     label_size,
                                     num_classes,
                                     is_training,
                                     device_str)
        self.num_features = [10]
        self.set_activation_type('relu')
        self.name = "ToyNet"
        print "{}\n"\
            "3x3x3 convolution {} kernels\n" \
            "Classifiying {} classes".format(
                self.name, self.num_features, self.num_classes)


    def inference(self, images, layer_id=None):
        BaseLayer._print_activations(images)
        print ""
        images = tf.expand_dims(images, 4)
        with tf.variable_scope('conv_1_1') as scope:
            conv_1_1 = self.conv_3x3(images, 1, self.num_features[0])
            conv_1_1 = self.batch_norm(conv_1_1)
            conv_1_1 = self.nonlinear_acti(conv_1_1)
            BaseLayer._print_activations(conv_1_1)
            print ""

        ## 1x1x1 convolution "fully connected"
        with tf.variable_scope('conv_fc') as scope:
            conv_fc = self.conv_layer_1x1(conv_1_1,
                                          self.num_features[0],
                                          self.num_classes,
                                          bn=True, acti=False)
            BaseLayer._print_activations(conv_fc)
            print ""

        if layer_id is None:
            return conv_fc
