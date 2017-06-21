# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from layer.base_layer import TrainableLayer
from layer.reshape import ReshapeLayer
from layer.fully_connected import FullyConnectedLayer
from layer.layer_util import infer_dims

class AutoEncoderBasic(TrainableLayer):
    """
        This is the most basic autoencoder, composed of a sequence of
        fully-connected layers.
        """

    def __init__(self,
                 num_classes=None,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='relu',
                 name='AE_basic'):

        super(AutoEncoderBasic, self).__init__(name=name)
        self.layer_sizes_encoder = [256, 128]
        self.acti_func_encoder = ['relu', 'relu']
        self.layer_sizes_decoder = [256]
        self.acti_func_decoder = ['relu']
        self.acti_func_output = 'identity'

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, images, is_training):

        [data_dimensions, data_dimensionality] = infer_dims(images)

        reshape_input = ReshapeLayer([-1, data_dimensionality])
        print(reshape_input)

        # Define the encoding layers
        encoders = []
        for i in range(0,len(self.layer_sizes_encoder)):
            encoders.append(FullyConnectedLayer(
                n_output_chns=self.layer_sizes_encoder[i],
                acti_func=self.acti_func_encoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='fc_encoder_{}'.format(self.layer_sizes_encoder[i])))
            print(encoders[-1])

        # Define the hidden decoding layers
        decoders = []
        for i in range(0, len(self.layer_sizes_decoder)):
            decoders.append(FullyConnectedLayer(
                n_output_chns=self.layer_sizes_decoder[i],
                acti_func=self.acti_func_decoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='fc_decoder_{}'.format(self.layer_sizes_decoder[i])))
            print(decoders[-1])

        # Define the output layer
        decoders.append(FullyConnectedLayer(
            n_output_chns=data_dimensionality,
            acti_func=self.acti_func_output,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='fc_decoder_{}'.format(data_dimensionality)))
        print(decoders[-1])

        reshape_output = ReshapeLayer([-1]+data_dimensions)
        print(reshape_output)

        flow = reshape_input(images)
        for i in range(0, len(self.layer_sizes_encoder)):
            flow = encoders[i](flow, is_training)
        codes = flow
        for i in range(0, len(self.layer_sizes_decoder)+1):
            flow = decoders[i](flow, is_training)
        reconstructions = flow
        reconstructions = reshape_output(reconstructions)

        return [images, reconstructions, codes]
