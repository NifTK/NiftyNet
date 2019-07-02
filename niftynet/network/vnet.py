# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer import layer_util
from niftynet.layer.activation import ActiLayer
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvLayer
from niftynet.layer.deconvolution import DeconvLayer
from niftynet.layer.elementwise import ElementwiseLayer
from niftynet.network.base_net import BaseNet
from niftynet.utilities.util_common import look_up_operations


class VNet(BaseNet):
    """
    ### Description
        implementation of V-Net:
          Milletari et al., "V-Net: Fully convolutional neural networks for
          volumetric medical image segmentation", 3DV '16

    ### Building Blocks
    (n)[dBLOCK]        - Downsampling VNet block with n conv layers (kernel size = 5,
                            with residual connections, activation = relu as default)
                            followed by downsampling conv layer (kernel size = 2,
                            stride = 2) + final activation
    (n)[uBLOCK]         - Upsampling VNet block with n conv layers (kernel size = 5,
                            with residual connections, activation = relu as default)
                            followed by deconv layer (kernel size = 2,
                            stride = 2) + final activation
    (n)[sBLOCK]         - VNet block with n conv layers (kernel size = 5,
                            with residual connections, activation = relu as default)
                            followed by 1x1x1 conv layer (kernel size = 1,
                            stride = 1) + final activation

    ### Diagram

    INPUT  -->  (1)[dBLOCK] - - - - - - - - - - - - - - - - (1)[sBLOCK] --> OUTPUT
                    |                                             |
                   (2)[dBLOCK] - - - - - - - - - - - - (2)[uBLOCK]
                        |                                   |
                        (3)[dBLOCK]  - - - - - - - (3)[uBLOCK]
                            |                           |
                            (3)[dBLOCK]  - - - (3)[uBLOCK]
                                |                   |
                                ----(3)[uBLOCk] ----


    ### Constraints
     - Input size should be divisible by 8
     - Input should be either 2D or 3D
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='VNet'):
        """

        :param num_classes: int, number of channels of output
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param acti_func: activation function to use
        :param name: layer name
        """

        super(VNet, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.n_features = [16, 32, 64, 128, 256]

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        """

        :param images: tensor to input to the network. Size has to be divisible by 8
        :param is_training:  boolean, True if network is in training mode
        :param layer_id: not in use
        :param unused_kwargs: other conditional arguments, not in use
        :return: tensor, network output
        """
        assert layer_util.check_spatial_dims(images, lambda x: x % 8 == 0)

        if layer_util.infer_spatial_rank(images) == 2:
            padded_images = tf.tile(images, [1, 1, 1, self.n_features[0]])
        elif layer_util.infer_spatial_rank(images) == 3:
            padded_images = tf.tile(images, [1, 1, 1, 1, self.n_features[0]])
        else:
            raise ValueError('not supported spatial rank of the input image')
        # downsampling  blocks
        res_1, down_1 = VNetBlock('DOWNSAMPLE', 1,
                                  self.n_features[0],
                                  self.n_features[1],
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  acti_func=self.acti_func,
                                  name='L1')(images, padded_images)
        res_2, down_2 = VNetBlock('DOWNSAMPLE', 2,
                                  self.n_features[1],
                                  self.n_features[2],
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  acti_func=self.acti_func,
                                  name='L2')(down_1, down_1)
        res_3, down_3 = VNetBlock('DOWNSAMPLE', 3,
                                  self.n_features[2],
                                  self.n_features[3],
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  acti_func=self.acti_func,
                                  name='L3')(down_2, down_2)
        res_4, down_4 = VNetBlock('DOWNSAMPLE', 3,
                                  self.n_features[3],
                                  self.n_features[4],
                                  acti_func=self.acti_func,
                                  name='L4')(down_3, down_3)
        # upsampling blocks
        _, up_4 = VNetBlock('UPSAMPLE', 3,
                            self.n_features[4],
                            self.n_features[4],
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='V_')(down_4, down_4)
        concat_r4 = ElementwiseLayer('CONCAT')(up_4, res_4)
        _, up_3 = VNetBlock('UPSAMPLE', 3,
                            self.n_features[4],
                            self.n_features[3],
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='R4')(concat_r4, up_4)
        concat_r3 = ElementwiseLayer('CONCAT')(up_3, res_3)
        _, up_2 = VNetBlock('UPSAMPLE', 3,
                            self.n_features[3],
                            self.n_features[2],
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='R3')(concat_r3, up_3)
        concat_r2 = ElementwiseLayer('CONCAT')(up_2, res_2)
        _, up_1 = VNetBlock('UPSAMPLE', 2,
                            self.n_features[2],
                            self.n_features[1],
                            w_initializer=self.initializers['w'],
                            w_regularizer=self.regularizers['w'],
                            acti_func=self.acti_func,
                            name='R2')(concat_r2, up_2)
        # final class score
        concat_r1 = ElementwiseLayer('CONCAT')(up_1, res_1)
        _, output_tensor = VNetBlock('SAME', 1,
                                     self.n_features[1],
                                     self.num_classes,
                                     w_initializer=self.initializers['w'],
                                     w_regularizer=self.regularizers['w'],
                                     b_initializer=self.initializers['b'],
                                     b_regularizer=self.regularizers['b'],
                                     acti_func=self.acti_func,
                                     name='R1')(concat_r1, up_1)
        return output_tensor


SUPPORTED_OP = set(['DOWNSAMPLE', 'UPSAMPLE', 'SAME'])


class VNetBlock(TrainableLayer):
    def __init__(self,
                 func,
                 n_conv,
                 n_feature_chns,
                 n_output_chns,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='vnet_block'):
        """

        :param func: string, defines final block operation (Downsampling, upsampling, same)
        :param n_conv: int, number of conv layers to apply
        :param n_feature_chns: int, number of feature channels (output channels) for each conv layer
        :param n_output_chns: int, number of output channels of the final block operation (func)
        :param w_initializer: weight initialisation of convolutional layers
        :param w_regularizer: weight regularisation of convolutional layers
        :param b_initializer: bias initialisation of convolutional layers
        :param b_regularizer: bias regularisation of convolutional layers
        :param acti_func: activation function to use
        :param name: layer name
        """

        super(VNetBlock, self).__init__(name=name)

        self.func = look_up_operations(func.upper(), SUPPORTED_OP)
        self.n_conv = n_conv
        self.n_feature_chns = n_feature_chns
        self.n_output_chns = n_output_chns
        self.acti_func = acti_func

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, main_flow, bypass_flow):
        """

        :param main_flow: tensor, input to the VNet block
        :param bypass_flow: tensor, input from skip connection
        :return: res_flow is tensor before final block operation (for residual connections),
            main_flow is final output tensor
        """
        for i in range(self.n_conv):
            main_flow = ConvLayer(name='conv_{}'.format(i),
                                  n_output_chns=self.n_feature_chns,
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  kernel_size=5)(main_flow)
            if i < self.n_conv - 1:  # no activation for the last conv layer
                main_flow = ActiLayer(
                    func=self.acti_func,
                    regularizer=self.regularizers['w'])(main_flow)
        res_flow = ElementwiseLayer('SUM')(main_flow, bypass_flow)

        if self.func == 'DOWNSAMPLE':
            main_flow = ConvLayer(
                name='downsample',
                n_output_chns=self.n_output_chns,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                kernel_size=2, stride=2, with_bias=True)(res_flow)
        elif self.func == 'UPSAMPLE':
            main_flow = DeconvLayer(
                name='upsample',
                n_output_chns=self.n_output_chns,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                kernel_size=2, stride=2, with_bias=True)(res_flow)
        elif self.func == 'SAME':
            main_flow = ConvLayer(name='conv_1x1x1',
                                  n_output_chns=self.n_output_chns,
                                  w_initializer=self.initializers['w'],
                                  w_regularizer=self.regularizers['w'],
                                  b_initializer=self.initializers['b'],
                                  b_regularizer=self.regularizers['b'],
                                  kernel_size=1, with_bias=True)(res_flow)
        main_flow = ActiLayer(self.acti_func)(main_flow)
        print(self)
        return res_flow, main_flow
