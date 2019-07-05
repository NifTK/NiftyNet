# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

# import niftynet.engine.logging as logging
import numpy as np
import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.deconvolution import DeconvolutionalLayer
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.layer.upsample import UpSampleLayer


class VAE(TrainableLayer):
    """
    ### Description
        This is a denoising, convolutional, variational autoencoder (VAE),
        composed of a sequence of {convolutions then downsampling} blocks,
        followed by a sequence of fully-connected layers,
        followed by a sequence of {transpose convolutions then upsampling} blocks.
        See Auto-Encoding Variational Bayes, Kingma & Welling, 2014.
        2DO: share the fully-connected parameters
        between the mean and logvar decoders.

    ### Building Blocks
    [ENCODER]               - See ConvEncoder class below
    [GAUSSIAN SAMPLER]      - See GaussianSampler class below
    [DECODER]               - See ConvDecoder class below

    ### Diagram

    INPUT --> [ENCODER] --> [GAUSSIAN SAMPLER] --> [FCDEC] ---> [DECODER] for means --- OUTPUTS
                                                        |                               |
                                                        ------> [DECODER] for logvars ---

    ### Constraints
    """

    def __init__(self,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='VAE'):
        """

        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param name: layer name
        """

        super(VAE, self).__init__(name=name)

        # The following options completely specify the model.
        # Note that the network need not be symmetric!

        # 1) Denoising
        # NOTES: If (and only if) 'denoising_variance' is
        # greater than zero, Gaussian noise with zero mean
        # and 'denoising_variance' variance is added to the input.
        self.denoising_variance = 0.001

        # 2) The convolutional layers
        # NOTES: the ith element of 'conv_pooling_factors' is
        # the amount of pooling in the ith layer
        # (along all spatial dimensions).
        # CONSTRAINTS: All four lists must be the same length.
        self.conv_output_channels = [32, 64, 96]
        self.conv_kernel_sizes = [3, 3, 3]
        self.conv_pooling_factors = [2, 2, 2]
        self.acti_func_conv = ['selu', 'selu', 'selu']

        # 3) The fully-connected layers
        # NOTES: If 'layer_sizes_decoder_shared' is empty then
        #  the data means and log variances are predicted
        # (separately) directly from the approximate sample
        # from the posterior. Otherwise, this sample is passed
        # through fully-connected layers of size
        # 'layer_sizes_decoder_shared', with respective activation
        # functions 'acti_func_decoder_shared'.
        # CONSTRAINTS:
        #   'layer_sizes_encoder' and 'acti_func_encoder'
        #   must have equal length.
        #   'acti_func_decoder' must be one element longer than
        #   'layer_sizes_decoder', because the final
        #   element of 'acti_func_decoder' is the activation function
        #   of the final fully-connected layer (which is added to the
        #   network automatically, and whose dimensionality equals
        #  that of the input to the fully-connected layers).

        self.layer_sizes_encoder = [256, 128]
        self.acti_func_encoder = ['selu', 'selu']
        self.number_of_latent_variables = 64
        self.number_of_samples_from_posterior = 100
        self.layer_sizes_decoder_shared = [128]
        self.acti_func_decoder_shared = ['selu']
        self.layer_sizes_decoder = self.layer_sizes_encoder[::-1]
        self.acti_func_decoder = self.acti_func_encoder[::-1] + ['selu']

        # 4) The transpose convolutional layers (for predicting means)
        # NOTES: 'upsampling_mode' determines how the feature maps
        #  in the decoding layers are upsampled.
        # The options are,
        # 1. 'DECONV' (recommended):
        #     kernel shape is HxWxDxChannelsInxChannelsOut,
        # 2. 'CHANNELWISE_DECONV':
        #     kernel shape is HxWxDx1x1,
        # 3. 'REPLICATE':
        #     no parameters.
        # CONSTRAINTS:
        #     'trans_conv_output_channels_means' is one element
        #     shorter than 'trans_conv_kernel_sizes_means',
        #     'trans_conv_unpooling_factors_means', and
        #     'trans_conv_unpooling_factors_means'
        #     because the final element of
        #     'trans_conv_output_channels_means' must be
        #     the number of channels in the input,
        #     and this is added to the list automatically.
        self.trans_conv_output_channels_means = \
            self.conv_output_channels[-2::-1]
        self.trans_conv_kernel_sizes_means = self.conv_kernel_sizes[::-1]
        self.trans_conv_unpooling_factors_means = \
            self.conv_pooling_factors[::-1]
        self.acti_func_trans_conv_means = \
            self.acti_func_conv[-2::-1] + ['sigmoid']
        self.upsampling_mode_means = 'DECONV'

        # 5) The transpose convolutional layers
        #     (for predicting (log) variances)
        # CONSTRAINTS:
        #     same as for the mean-predicting layers above.
        self.trans_conv_output_channels_logvars = \
            self.trans_conv_output_channels_means
        self.trans_conv_kernel_sizes_logvars = \
            self.trans_conv_kernel_sizes_means
        self.trans_conv_unpooling_factors_logvars = \
            self.trans_conv_unpooling_factors_means
        self.acti_func_trans_conv_logvars = \
            self.acti_func_conv[-2::-1] + [None]
        self.upsampling_mode_logvars = self.upsampling_mode_means

        # 6) Clip logvars to avoid infs & nans
        # NOTES: variance = exp(logvars),
        # so we must keep logvars within reasonable limits.
        self.logvars_upper_bound = 50
        self.logvars_lower_bound = -self.logvars_upper_bound

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, images, is_training=True, **unused_kwargs):
        """

        :param images: tensor, input to the network
        :param is_training: boolean, True if network is in training mode
        :param unused_kwargs: other conditional arguments, not in use
        :return: posterior_means: means output by gaussian sampler for KL divergence
                posterior_logvars: log of variances output by gaussian sampler for KL divergence
                data_means: output of means decoder branch (predicted image)
                data_logvars: output of variances decoder branch (capturing aleatoric uncertainty)
                images: input
                data_variances: exp of data_logvars
                posterior_variances: exp of posterior_logvars
                sample: random sample from latent space (from gaussian sampler)
        """

        def clip(x):
            """
            Clip input tensor using the lower and upper bounds
            :param x: tensor, input
            :return: clipped tensor
            """
            return tf.clip_by_value(x, self.logvars_lower_bound,
                                    self.logvars_upper_bound)

        def normalise(x):
            """
            Normalise input to [0, 255]
            :param x: tensor, input to normalise
            :return: normalised tensor in [0, 255]
            """
            min_val = tf.reduce_min(x)
            max_val = tf.reduce_max(x)
            return 255 * (x - min_val) / (max_val - min_val)

        def infer_downsampled_shape(x, output_channels, pooling_factors):
            """
            Calculate the shape of the data as it emerges from
            the convolutional part of the encoder
            :param x: tensor, input
            :param output_channels: int, number of output channels
            :param pooling_factors: array, pooling factors
            :return: array, shape of downsampled image
            """
            downsampled_shape = x.shape[1::].as_list()
            downsampled_shape[-1] = output_channels[-1]
            downsampled_shape[0:-1] = \
                downsampled_shape[0:-1] / np.prod(pooling_factors)
            return [int(x) for x in downsampled_shape]

        # Derive shape information from the input
        input_shape = images.shape[1::].as_list()
        number_of_input_channels = input_shape[-1]
        downsampled_shape = infer_downsampled_shape(images,
                                                    self.conv_output_channels,
                                                    self.conv_pooling_factors)
        serialised_shape = int(np.prod(downsampled_shape))

        encoder = ConvEncoder(self.denoising_variance,
                              self.conv_output_channels,
                              self.conv_kernel_sizes,
                              self.conv_pooling_factors,
                              self.acti_func_conv,
                              self.layer_sizes_encoder,
                              self.acti_func_encoder,
                              serialised_shape)

        approximate_sampler = GaussianSampler(
            self.number_of_latent_variables,
            self.number_of_samples_from_posterior,
            self.logvars_upper_bound,
            self.logvars_lower_bound)

        # Initialise the shared fully-connected layers,
        # if and only if they have been specified
        if len(self.layer_sizes_decoder_shared) > 0:
            self.shared_decoder = FCDecoder(self.layer_sizes_decoder_shared,
                                       self.acti_func_decoder_shared,
                                       name='FCDecoder')

        self.decoder_means = ConvDecoder(
            self.layer_sizes_decoder + [serialised_shape],
            self.acti_func_decoder,
            self.trans_conv_output_channels_means + [number_of_input_channels],
            self.trans_conv_kernel_sizes_means,
            self.trans_conv_unpooling_factors_means,
            self.acti_func_trans_conv_means,
            self.upsampling_mode_means,
            downsampled_shape,
            name='ConvDecoder_means')

        decoder_logvars = ConvDecoder(
            self.layer_sizes_decoder + [serialised_shape],
            self.acti_func_decoder,
            self.trans_conv_output_channels_logvars + [
                number_of_input_channels],
            self.trans_conv_kernel_sizes_logvars,
            self.trans_conv_unpooling_factors_logvars,
            self.acti_func_trans_conv_logvars,
            self.upsampling_mode_logvars,
            downsampled_shape,
            name='ConvDecoder_logvars')

        # Encode the input
        encoding = encoder(images, is_training)

        # Sample from the posterior distribution P(latent variables|input)
        [sample, posterior_means, posterior_logvars] = approximate_sampler(
            encoding, is_training)

        if len(self.layer_sizes_decoder_shared) > 0:
            partially_decoded_sample = self.shared_decoder(
                sample, is_training)
        else:
            partially_decoded_sample = sample

        [data_means, data_logvars] = [
            self.decoder_means(partially_decoded_sample, is_training),
            clip(decoder_logvars(sample, is_training))]

        ## Monitor the KL divergence of
        ## the (approximate) posterior from the prior
        #KL_divergence = 1 + posterior_logvars \
        #                - tf.square(posterior_means) \
        #                - tf.exp(posterior_logvars)
        #KL_divergence = -0.5 * tf.reduce_mean(
        #    tf.reduce_sum(KL_divergence, axis=[1]))
        ## tf.add_to_collection(
        ##     logging.CONSOLE, tf.summary.scalar('KL_divergence', KL_divergence))

        ## Monitor the (negative log) likelihood of the parameters given the data
        #log_likelihood = data_logvars + \
        #                 np.log(2 * np.pi) + \
        #                 tf.exp(-data_logvars) * tf.square(data_means - images)
        #log_likelihood = -0.5 * tf.reduce_mean(tf.reduce_sum(
        #    log_likelihood, axis=[1, 2, 3, 4]))
        ## tf.add_to_collection(
        ##     logging.CONSOLE,
        ##     tf.summary.scalar('negative_log_likelihood', -log_likelihood))

        posterior_variances = tf.exp(posterior_logvars)
        data_variances = tf.exp(data_logvars)

        # Monitor reconstructions
        # logging.image3_coronal('Originals', normalise(images))
        # logging.image3_coronal('Means', normalise(data_means))
        # logging.image3_coronal('Variances', normalise(data_variances))

        return [posterior_means,
                posterior_logvars,
                data_means,
                data_logvars,
                images,
                data_variances,
                posterior_variances,
                sample]


class ConvEncoder(TrainableLayer):
    """
    ### Description
        This is a generic encoder composed of
        {convolutions then downsampling} blocks followed by
        fully-connected layers.
    """

    def __init__(self,
                 denoising_variance,
                 conv_output_channels,
                 conv_kernel_sizes,
                 conv_pooling_factors,
                 acti_func_conv,
                 layer_sizes_encoder,
                 acti_func_encoder,
                 serialised_shape,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='ConvEncoder'):
        """

        :param denoising_variance: variance of gaussian noise to add to the input
        :param conv_output_channels: array, number of output channels for each conv layer
        :param conv_kernel_sizes: array, kernel sizes for each conv layer
        :param conv_pooling_factors: array, stride values for downsampling convolutions
        :param acti_func_conv: array, activation functions of each layer
        :param layer_sizes_encoder: array, number of output channels for each encoding FC layer
        :param acti_func_encoder: array, activation functions for each encoding FC layer
        :param serialised_shape: array, flatten shape to enter the FC layers
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param name: layer name
        """

        super(ConvEncoder, self).__init__(name=name)

        self.denoising_variance = denoising_variance
        self.conv_output_channels = conv_output_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pooling_factors = conv_pooling_factors
        self.acti_func_conv = acti_func_conv
        self.layer_sizes_encoder = layer_sizes_encoder
        self.acti_func_encoder = acti_func_encoder
        self.serialised_shape = serialised_shape

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, images, is_training):
        """

        :param images: tensor, input to the network
        :param is_training: boolean, True if network is in training mode
        :return: tensor, output of the encoder branch
        """

        # Define the encoding convolutional layers
        encoders_cnn = []
        encoders_downsamplers = []
        for i in range(0, len(self.conv_output_channels)):
            encoders_cnn.append(ConvolutionalLayer(
                n_output_chns=self.conv_output_channels[i],
                kernel_size=self.conv_kernel_sizes[i],
                padding='SAME',
                with_bias=True,
                feature_normalization='batch',
                w_initializer=self.initializers['w'],
                w_regularizer=None,
                acti_func=self.acti_func_conv[i],
                name='encoder_conv_{}_{}'.format(
                    self.conv_kernel_sizes[i],
                    self.conv_output_channels[i])))
            print(encoders_cnn[-1])

            encoders_downsamplers.append(ConvolutionalLayer(
                n_output_chns=self.conv_output_channels[i],
                kernel_size=self.conv_pooling_factors[i],
                stride=self.conv_pooling_factors[i],
                padding='SAME',
                with_bias=False,
                feature_normalization='batch',
                w_initializer=self.initializers['w'],
                w_regularizer=None,
                acti_func=self.acti_func_conv[i],
                name='encoder_downsampler_{}_{}'.format(
                    self.conv_pooling_factors[i],
                    self.conv_pooling_factors[i])))

            print(encoders_downsamplers[-1])

        # Define the encoding fully-connected layers
        encoders_fc = []
        for i in range(0, len(self.layer_sizes_encoder)):
            encoders_fc.append(FullyConnectedLayer(
                n_output_chns=self.layer_sizes_encoder[i],
                with_bias=True,
                feature_normalization='batch',
                acti_func=self.acti_func_encoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='encoder_fc_{}'.format(self.layer_sizes_encoder[i])))
            print(encoders_fc[-1])

        # Add Gaussian noise to the input
        if self.denoising_variance > 0 and is_training:
            flow = images + tf.random_normal(
                    tf.shape(images), 0.0, self.denoising_variance)
        else:
            flow = images

        # Convolutional encoder layers
        for i in range(0, len(self.conv_output_channels)):
            flow = encoders_downsamplers[i](
                encoders_cnn[i](flow, is_training), is_training)

        # Flatten the feature maps
        flow = tf.reshape(flow, [-1, self.serialised_shape])

        # Fully-connected encoder layers
        for i in range(0, len(self.layer_sizes_encoder)):
            flow = encoders_fc[i](flow, is_training)

        return flow


class GaussianSampler(TrainableLayer):
    """
    ### Description
        This predicts the mean and logvariance parameters,
        then generates an approximate sample from the posterior.

    ### Diagram

    ### Constraints

    """

    def __init__(self,
                 number_of_latent_variables,
                 number_of_samples_from_posterior,
                 logvars_upper_bound,
                 logvars_lower_bound,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='gaussian_sampler'):
        """

        :param number_of_latent_variables: int, number of output channels for FC layer
        :param number_of_samples_from_posterior: int, number of samples to draw from standard gaussian
        :param logvars_upper_bound: upper bound of log of variances for clipping
        :param logvars_lower_bound: lower bound of log of variances for clipping
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param name: layer name
        """

        super(GaussianSampler, self).__init__(name=name)

        self.number_of_latent_variables = number_of_latent_variables
        self.number_of_samples = number_of_samples_from_posterior
        self.logvars_upper_bound = logvars_upper_bound
        self.logvars_lower_bound = logvars_lower_bound

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, codes, is_training):
        """

        :param codes: tensor, input latent space
        :param is_training: boolean, True if network is in training mode
        :return: samples from posterior distribution, means and log variances of the posterior distribution
        """

        def clip(input):
            # This is for clipping logvars,
            # so that variances = exp(logvars) behaves well
            output = tf.maximum(input, self.logvars_lower_bound)
            output = tf.minimum(output, self.logvars_upper_bound)
            return output

        encoder_means = FullyConnectedLayer(
            n_output_chns=self.number_of_latent_variables,
            feature_normalization=None,
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='encoder_fc_means_{}'.format(self.number_of_latent_variables))
        print(encoder_means)

        encoder_logvars = FullyConnectedLayer(
            n_output_chns=self.number_of_latent_variables,
            feature_normalization=None,
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='encoder_fc_logvars_{}'.format(
                self.number_of_latent_variables))
        print(encoder_logvars)

        # Predict the posterior distribution's parameters
        posterior_means = encoder_means(codes, is_training)
        posterior_logvars = clip(encoder_logvars(codes, is_training))

        if self.number_of_samples == 1:
            noise_sample = tf.random_normal(tf.shape(posterior_means),
                                            0.0,
                                            1.0)
        else:
            sample_shape = tf.concat(
                [tf.constant(self.number_of_samples, shape=[1, ]),
                 tf.shape(posterior_means)], axis=0)
            noise_sample = tf.reduce_mean(
                tf.random_normal(sample_shape, 0.0, 1.0), axis=0)

        return [
            posterior_means + tf.exp(0.5 * posterior_logvars) * noise_sample,
            posterior_means, posterior_logvars]


class ConvDecoder(TrainableLayer):
    """
    ### Description
            This is a generic decoder composed of
            fully-connected layers followed by
            {upsampling then transpose convolution} blocks.
            There is no batch normalisation on
            the final transpose convolutional layer.
    """

    def __init__(self,
                 layer_sizes_decoder,
                 acti_func_decoder,
                 trans_conv_output_channels,
                 trans_conv_kernel_sizes,
                 trans_conv_unpooling_factors,
                 acti_func_trans_conv,
                 upsampling_mode,
                 downsampled_shape,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='ConvDecoder'):
        """

        :param layer_sizes_decoder: array, number of output channels for each decoding FC layer
        :param acti_func_decoder: array, activation functions for each decoding FC layer
        :param trans_conv_output_channels: array, number of output channels for each transpose conv layer
        :param trans_conv_kernel_sizes: array, kernel sizes for each transpose conv layer
        :param trans_conv_unpooling_factors: array, stride values for upsampling transpose convolutions
        :param acti_func_trans_conv: array, activation functions for each transpose conv layer
        :param upsampling_mode: string, type of upsampling (Deconvolution, channelwise deconvolution, replicate)
        :param downsampled_shape: array, final encoded shape before FC in encoder
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param name: layer name
        """

        super(ConvDecoder, self).__init__(name=name)

        self.layer_sizes_decoder = layer_sizes_decoder
        self.acti_func_decoder = acti_func_decoder
        self.trans_conv_output_channels = trans_conv_output_channels
        self.trans_conv_kernel_sizes = trans_conv_kernel_sizes
        self.trans_conv_unpooling_factors = trans_conv_unpooling_factors
        self.acti_func_trans_conv = acti_func_trans_conv
        self.upsampling_mode = upsampling_mode
        self.downsampled_shape = downsampled_shape

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, codes, is_training):
        """

        :param codes: tensor, input latent space after gaussian sampling
        :param is_training: boolean, True if network is in training mode
        :return: tensor, output of decoding branch
        """

        # Define the decoding fully-connected layers
        decoders_fc = []
        for i in range(0, len(self.layer_sizes_decoder)):
            decoders_fc.append(FullyConnectedLayer(
                n_output_chns=self.layer_sizes_decoder[i],
                with_bias=True,
                feature_normalization='batch',
                acti_func=self.acti_func_decoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='decoder_fc_{}'.format(self.layer_sizes_decoder[i])))
            print(decoders_fc[-1])

        # Define the decoding convolutional layers
        decoders_cnn = []
        decoders_upsamplers = []
        for i in range(0, len(self.trans_conv_output_channels)):
            if self.upsampling_mode == 'DECONV':
                decoders_upsamplers.append(DeconvolutionalLayer(
                    n_output_chns=self.trans_conv_output_channels[i],
                    kernel_size=self.trans_conv_unpooling_factors[i],
                    stride=self.trans_conv_unpooling_factors[i],
                    padding='SAME',
                    with_bias=True,
                    feature_normalization='batch',
                    w_initializer=self.initializers['w'],
                    w_regularizer=None,
                    acti_func=None,
                    name='decoder_upsampler_{}_{}'.format(
                        self.trans_conv_unpooling_factors[i],
                        self.trans_conv_unpooling_factors[i])))
                print(decoders_upsamplers[-1])

            decoders_cnn.append(DeconvolutionalLayer(
                n_output_chns=self.trans_conv_output_channels[i],
                kernel_size=self.trans_conv_kernel_sizes[i],
                stride=1,
                padding='SAME',
                with_bias=True,
                feature_normalization='batch',
                #feature_normalization=not (i == len(self.trans_conv_output_channels) - 1),
                # No BN on output
                w_initializer=self.initializers['w'],
                w_regularizer=None,
                acti_func=self.acti_func_trans_conv[i],
                name='decoder_trans_conv_{}_{}'.format(
                    self.trans_conv_kernel_sizes[i],
                    self.trans_conv_output_channels[i])))
            print(decoders_cnn[-1])

        # Fully-connected decoder layers
        flow = codes
        for i in range(0, len(self.layer_sizes_decoder)):
            flow = decoders_fc[i](flow, is_training)

        # Reconstitute the feature maps
        flow = tf.reshape(flow, [-1] + self.downsampled_shape)

        # Convolutional decoder layers
        for i in range(0, len(self.trans_conv_output_channels)):
            if self.upsampling_mode == 'DECONV':
                flow = decoders_upsamplers[i](flow, is_training)
            elif self.upsampling_mode == 'CHANNELWISE_DECONV':
                flow = UpSampleLayer(
                    'CHANNELWISE_DECONV',
                    kernel_size=self.trans_conv_unpooling_factors[i],
                    stride=self.trans_conv_unpooling_factors[i])(flow)
            elif self.upsampling_mode == 'REPLICATE':
                flow = UpSampleLayer(
                    'REPLICATE',
                    kernel_size=self.trans_conv_unpooling_factors[i],
                    stride=self.trans_conv_unpooling_factors[i])(flow)
            flow = decoders_cnn[i](flow, is_training)

        return flow


class FCDecoder(TrainableLayer):
    """
    ### Description
        This is a generic fully-connected decoder.
    """

    def __init__(self,
                 layer_sizes_decoder,
                 acti_func_decoder,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='FCDecoder'):
        """

        :param layer_sizes_decoder: array, number of output channels for each decoding FC layer
        :param acti_func_decoder: array, activation functions for each decoding FC layer
        :param w_initializer: weight initialisation for network
        :param w_regularizer: weight regularisation for network
        :param b_initializer: bias initialisation for network
        :param b_regularizer: bias regularisation for network
        :param name: layer name
        """

        super(FCDecoder, self).__init__(name=name)

        self.layer_sizes_decoder = layer_sizes_decoder
        self.acti_func_decoder = acti_func_decoder

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, codes, is_training):
        """

        :param codes: tensor, input latent codes
        :param is_training: boolean, True if network is in training mode
        :return: tensor, output of series of FC layers
        """

        # Define the decoding fully-connected layers
        decoders_fc = []
        for i in range(0, len(self.layer_sizes_decoder)):
            decoders_fc.append(FullyConnectedLayer(
                n_output_chns=self.layer_sizes_decoder[i],
                with_bias=True,
                feature_normalization='batch',
                acti_func=self.acti_func_decoder[i],
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                name='FCDecoder_fc_{}'.format(self.layer_sizes_decoder[i])))
            print(decoders_fc[-1])

        # Fully-connected decoder layers
        flow = codes
        for i in range(0, len(self.layer_sizes_decoder)):
            flow = decoders_fc[i](flow, is_training)

        return flow
