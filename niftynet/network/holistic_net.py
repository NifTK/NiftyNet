# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.upsample import UpSampleLayer
from niftynet.network.base_net import BaseNet
from niftynet.network.highres3dnet import HighResBlock


class HolisticNet(BaseNet):
    """
    ### Description
    Implementation of HolisticNet detailed in
    Fidon, L. et. al. (2017) Generalised Wasserstein Dice Score for Imbalanced
    Multi-class Segmentation using Holistic Convolutional Networks.
    MICCAI 2017 (BrainLes)

    ### Diagram Blocks
    [CONV]          -   3x3x3 Convolutional layer in form: Activation(Convolution(X))
                        where X = input tensor or output of previous layer

                        and Activation is a function which includes:

                            a) Batch-Norm
                            b) Activation Function (Elu, ReLu, PreLu, Sigmoid, Tanh etc.)

    [D-CONV(d)]     -   3x3x3 Convolutional layer with dilated convolutions with blocks in
                         pre-activation mode: D-Convolution(Activation(X))
                         see He et al., "Identity Mappings in Deep Residual Networks", ECCV '16

                         dilation factor = d
                         D-CONV(2) : dilated convolution with dilation factor 2

                         repeat factor = r
                         e.g.
                         (2)[D-CONV(d)]     : 2 dilated convolutional layers in a row [D-CONV] -> [D-CONV]
                         { (2)[D-CONV(d)] } : 2 dilated convolutional layers within residual block

    [SCORE]         -   Batch-Norm + 3x3x3 Convolutional layer  + Activation function + 1x1x1 Convolutional layer

    [MERGE]         -   Channel-wise merging


    ### Diagram

    MULTIMODAL INPUT ----- [CONV]x3 -----[D-CONV(2)]x3 ----- MaxPooling ----- [CONV]x3 -----[D-CONV(2)]x3
                                        |                   |                          |                  |
                                      [SCORE]             [SCORE]                    [SCORE]            [SCORE]
                                        |                   |                          |                  |
                                        -------------------------------------------------------------------
                                                                        |
                                                                     [MERGE] --> OUTPUT

    ### Constraints
    - Input image size should be divisible by 8

    ### Comments
    - The network returns only the merged output, so the loss will be applied only to this
    (different from the referenced paper)
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='elu',
                 name='HolisticNet'):
        """

        :param num_classes: int, number of channels of output
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param acti_func: activation function to use
        :param name: layer name
        """
        super(HolisticNet, self).__init__(
            num_classes=num_classes,
            acti_func=acti_func,
            name=name,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer)

        self.num_res_blocks = [3, 3, 3, 3]
        self.num_features = [70] * 4
        self.num_fea_score_layers = [[70, 140]] * 4

        # self.loss = LossFunction(num_classes, loss_type='Dice', decay=0.0)

    def layer_op(self,
                 input_tensor,
                 is_training=True,
                 layer_id=-1,
                 **unused_kwargs):
        """

        :param input_tensor: tensor, input to the network
        :param is_training: boolean, True if network is in training mode
        :param layer_id: not in use
        :param unused_kwargs:
        :return: fused prediction from multiple scales
        """
        layer_instances = []
        scores_instances = []
        first_conv_layer = ConvolutionalLayer(
            n_output_chns=self.num_features[0],
            feature_normalization='batch',
            kernel_size=3,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            acti_func=self.acti_func,
            name='conv_1_1')
        flow = first_conv_layer(input_tensor, is_training)
        layer_instances.append((first_conv_layer, flow))

        # SCALE 1
        with DilatedTensor(flow, dilation_factor=1) as dilated:
            for j in range(self.num_res_blocks[0]):
                res_block = HighResBlock(
                    self.num_features[0],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % ('res_1', j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        score_layer_scale1 = ScoreLayer(
            num_features=self.num_fea_score_layers[0],
            num_classes=self.num_classes)
        score_1 = score_layer_scale1(flow, is_training)
        scores_instances.append(score_1)
        # if is_training:
        #     loss_s1 = WGDL(score_1, labels)
        #     tf.add_to_collection('multiscale_loss', loss_s1/num_scales)

        # # SCALE 2
        with DilatedTensor(flow, dilation_factor=2) as dilated:
            for j in range(self.num_res_blocks[1]):
                res_block = HighResBlock(
                    self.num_features[1],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % ('res_2', j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        score_layer_scale2 = ScoreLayer(
            num_features=self.num_fea_score_layers[1],
            num_classes=self.num_classes)
        score_2 = score_layer_scale2(flow, is_training)

        # score_2 = self.score_layer(flow, self.num_fea_score_layers[1])
        up_score_2 = score_2
        scores_instances.append(up_score_2)
        # if is_training:
        #     loss_s2 =  self.WGDL(score_2, labels)
        #     # loss_s2 = self.new_dice_loss(score_2, labels)
        #     tf.add_to_collection('multiscale_loss', loss_s2/num_scales)


        # SCALE 3
        ## dowsampling factor = 2
        downsample_scale3 = DownSampleLayer(
            func='AVG', kernel_size=2, stride=2)
        flow = downsample_scale3(flow)
        layer_instances.append((downsample_scale3, flow))
        with DilatedTensor(flow, dilation_factor=1) as dilated:
            for j in range(self.num_res_blocks[2]):
                res_block = HighResBlock(
                    self.num_features[2],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % ('res_3', j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        score_layer_scale3 = ScoreLayer(
            num_features=self.num_fea_score_layers[2],
            num_classes=self.num_classes)
        score_3 = score_layer_scale3(flow, is_training)

        upsample_indep_scale3 = UpSampleLayer(
            func='CHANNELWISE_DECONV',
            kernel_size=2,
            stride=2,
            w_initializer=tf.constant_initializer(1.0, dtype=tf.float32))
        up_score_3 = upsample_indep_scale3(score_3)
        scores_instances.append(up_score_3)

        # up_score_3 = self.feature_indep_upsample_conv(score_3, factor=2)
        # if is_training:
        #     loss_s3 = self.WGDL(up_score_3, labels)
        #     # loss_s3 = self.new_dice_loss(up_score_3, labels)
        #     tf.add_to_collection('multiscale_loss', loss_s3/num_scales)

        # SCALE 4
        with DilatedTensor(flow, dilation_factor=2) as dilated:
            for j in range(self.num_res_blocks[3]):
                res_block = HighResBlock(
                    self.num_features[3],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % ('res_4', j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        score_layer_scale4 = ScoreLayer(
            num_features=self.num_fea_score_layers[3],
            num_classes=self.num_classes)
        score_4 = score_layer_scale4(
            flow,
            self.num_fea_score_layers[3],
            is_training)

        upsample_indep_scale4 = UpSampleLayer(
            func='CHANNELWISE_DECONV',
            kernel_size=1,
            stride=2,
            w_initializer=tf.constant_initializer(1.0, dtype=tf.float32))
        up_score_4 = upsample_indep_scale4(score_4)
        scores_instances.append(up_score_4)

        # if is_training:
        #     loss_s4 = self.WGDL(up_score_4, labels)
        #     # loss_s4 = self.new_dice_loss(up_score_4, labels)
        #     tf.add_to_collection('multiscale_loss', loss_s4/num_scales)

        # FUSED SCALES
        merge_layer = MergeLayer('WEIGHTED_AVERAGE')
        soft_scores = []
        for s in scores_instances:
            soft_scores.append(tf.nn.softmax(s))
        fused_score = merge_layer(soft_scores)
        scores_instances.append(fused_score)
        if is_training:
            return scores_instances
        return fused_score


class ScoreLayer(TrainableLayer):
    def __init__(self,
                 num_features=None,
                 w_initializer=None,
                 w_regularizer=None,
                 num_classes=1,
                 acti_func='elu',
                 name='ScoreLayer'):
        """

        :param num_features: int, number of features
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param num_classes: int, number of prediction channels
        :param acti_func: activation function to use
        :param name: layer name
        """
        super(ScoreLayer, self).__init__(name=name)
        self.num_classes = num_classes
        self.acti_func = acti_func
        self.num_features = num_features
        self.n_layers = len(self.num_features)
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor, is_training, layer_id=-1):
        """

        :param input_tensor: tensor, input to the layer
        :param is_training: boolean, True if network is in training mode
        :param layer_id: not is use
        :return: tensor with number of channels to num_classes
        """
        rank = input_tensor.shape.ndims
        perm = [i for i in range(rank)]
        perm[-2], perm[-1] = perm[-1], perm[-2]
        output_tensor = input_tensor
        n_layers = self.n_layers
        # All layers except the last one consists in:
        # BN + Conv_3x3x3 + Activation
        # layer_instances = []

        for layer in range(n_layers - 1):
            layer_to_add = ConvolutionalLayer(
                n_output_chns=self.num_features[layer + 1],
                feature_normalization='batch',
                kernel_size=3,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func=self.acti_func,
                name='conv_fc_%d' % layer)
            output_tensor = layer_to_add(output_tensor, is_training)
            # layer_instances.append((layer_to_add, output_tensor))
        last_layer = ConvolutionalLayer(n_output_chns=self.num_classes,
                                        kernel_size=1)
        output_tensor = last_layer(output_tensor, is_training)
        # layer_instances.append((last_layer, output_tensor))
        return output_tensor


SUPPORTED_OPS = set(['AVERAGE', 'WEIGHTED_AVERAGE', 'MAXOUT'])


class MergeLayer(TrainableLayer):
    def __init__(self,
                 func,
                 w_initializer=None,
                 w_regularizer=None,
                 acti_func='elu',
                 name='MergeLayer'):
        """

        :param func: type of merging layer (SUPPORTED_OPS: AVERAGE, WEIGHTED_AVERAGE, MAXOUT)
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param acti_func: activation function to use
        :param name: layer name
        """
        super(MergeLayer, self).__init__(name=name)
        self.func = func
        self.acti_func = acti_func
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, roots):
        """
        Performs channel-wise merging of input tensors
        :param roots: tensors to be merged
        :return: fused tensor
        """
        if self.func == 'MAXOUT':
            return tf.reduce_max(tf.stack(roots, axis=-1), axis=-1)
        elif self.func == 'AVERAGE':
            return tf.reduce_mean(tf.stack(roots, axis=-1), axis=-1)
        elif self.func == 'WEIGHTED_AVERAGE':
            input_tensor = tf.stack(roots, axis=-1)
            rank = input_tensor.shape.ndims
            perm = [i for i in range(rank)]
            perm[-2], perm[-1] = perm[-1], perm[-2]

            output_tensor = input_tensor
            output_tensor = tf.transpose(output_tensor, perm=perm)
            output_tensor = tf.unstack(output_tensor, axis=-1)
            roots_merged = []
            for f in range(len(output_tensor)):
                conv_layer = ConvLayer(
                    n_output_chns=1, kernel_size=1, stride=1)
                roots_merged_f = conv_layer(output_tensor[f])
                roots_merged.append(roots_merged_f)
            return tf.concat(roots_merged, axis=-1)
