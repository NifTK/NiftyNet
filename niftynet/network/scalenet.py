# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import tensorflow as tf
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.network.base_net import BaseNet
from niftynet.network.highres3dnet import HighRes3DNet, HighResBlock
from niftynet.utilities.util_common import look_up_operations


class ScaleNet(BaseNet):
    """
    implementation of ScaleNet:
        Fidon et al., "Scalable multimodal convolutional
        networks for brain tumour segmentation", MICCAI '17

    ### Diagram

    INPUT --> [BACKEND] ----> [MERGING] ----> [FRONTEND] ---> OUTPUT

    [BACKEND] and [MERGING] are provided by the ScaleBlock below
    [FRONTEND]: it can be any NiftyNet network (default: HighRes3dnet)

    ### Constraints:
    - Input image size should be divisible by 8
    - more than one modality should be used
    """

    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='prelu',
                 name='ScaleNet'):
        """

        :param num_classes: int, number of channels of output
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param acti_func: activation function to use
        :param name: layer name
        """

        super(ScaleNet, self).__init__(
            num_classes=num_classes,
            w_initializer=w_initializer,
            w_regularizer=w_regularizer,
            b_initializer=b_initializer,
            b_regularizer=b_regularizer,
            acti_func=acti_func,
            name=name)

        self.n_features = 16

    def layer_op(self, images, is_training=True, layer_id=-1, **unused_kwargs):
        """

        :param images: tensor, concatenation of multiple input modalities
        :param is_training: boolean, True if network is in training mode
        :param layer_id: not in use
        :param unused_kwargs:
        :return: predicted tensor
        """
        n_modality = images.shape.as_list()[-1]
        rank = images.shape.ndims
        assert n_modality > 1
        roots = tf.split(images, n_modality, axis=rank - 1)
        for (idx, root) in enumerate(roots):
            conv_layer = ConvolutionalLayer(
                n_output_chns=self.n_features,
                kernel_size=3,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func=self.acti_func,
                name='conv_{}'.format(idx))
            roots[idx] = conv_layer(root, is_training)
        roots = tf.stack(roots, axis=-1)

        back_end = ScaleBlock('AVERAGE', n_layers=1)
        output_tensor = back_end(roots, is_training)

        front_end = HighRes3DNet(self.num_classes)
        output_tensor = front_end(output_tensor, is_training)
        return output_tensor


SUPPORTED_OP = set(['MAX', 'AVERAGE'])


class ScaleBlock(TrainableLayer):
    """
    Implementation of the ScaleBlock described in
    Fidon et al., "Scalable multimodal convolutional
        networks for brain tumour segmentation", MICCAI '17

    See Fig 2(a) for diagram details - SN BackEnd

    """
    def __init__(self,
                 func,
                 n_layers=1,
                 w_initializer=None,
                 w_regularizer=None,
                 acti_func='relu',
                 name='scaleblock'):
        """
        :param func: merging function (SUPPORTED_OP: MAX, AVERAGE)
        :param n_layers: int, number of layers
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param acti_func: activation function to use
        :param name: layer name
        """
        self.func = look_up_operations(func.upper(), SUPPORTED_OP)
        super(ScaleBlock, self).__init__(name=name)
        self.n_layers = n_layers
        self.acti_func = acti_func

        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor, is_training):
        """

        :param input_tensor: tensor, input to the network
        :param is_training: boolean, True if network is in training mode
        :return: merged tensor after backend layers
        """
        n_modality = input_tensor.shape.as_list()[-1]
        n_chns = input_tensor.shape.as_list()[-2]
        rank = input_tensor.shape.ndims
        perm = [i for i in range(rank)]
        perm[-2], perm[-1] = perm[-1], perm[-2]

        output_tensor = input_tensor
        for layer in range(self.n_layers):
            # modalities => feature channels
            output_tensor = tf.transpose(output_tensor, perm=perm)
            output_tensor = tf.unstack(output_tensor, axis=-1)
            for (idx, tensor) in enumerate(output_tensor):
                block_name = 'M_F_{}_{}'.format(layer, idx)
                highresblock_op = HighResBlock(
                    n_output_chns=n_modality,
                    kernels=(3, 1),
                    with_res=True,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    acti_func=self.acti_func,
                    name=block_name)
                output_tensor[idx] = highresblock_op(tensor, is_training)
                print(highresblock_op)
            output_tensor = tf.stack(output_tensor, axis=-1)

            # feature channels => modalities
            output_tensor = tf.transpose(output_tensor, perm=perm)
            output_tensor = tf.unstack(output_tensor, axis=-1)
            for (idx, tensor) in enumerate(output_tensor):
                block_name = 'F_M_{}_{}'.format(layer, idx)
                highresblock_op = HighResBlock(
                    n_output_chns=n_chns,
                    kernels=(3, 1),
                    with_res=True,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    acti_func=self.acti_func,
                    name=block_name)
                output_tensor[idx] = highresblock_op(tensor, is_training)
                print(highresblock_op)
            output_tensor = tf.stack(output_tensor, axis=-1)

        if self.func == 'MAX':
            output_tensor = tf.reduce_max(output_tensor, axis=-1)
        elif self.func == 'AVERAGE':
            output_tensor = tf.reduce_mean(output_tensor, axis=-1)
        return output_tensor
