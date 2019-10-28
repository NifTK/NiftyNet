# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.downsample import DownSampleLayer


class SubPixelLayer(TrainableLayer):
    """
    Implementation of:

    SubPixel Convolution initialised with ICNR initialisation
    and followed by an AveragePooling

    Limitations:

    If ICNR initialization is used then the upsample factor
    MUST be an integer

    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Shi, W., Caballero, J., Huszár, F., Totz, J., Aitken, A.P.,
    Bishop, R., Rueckert, D. and Wang, Z., 2016.

    Real-time single image and video super-resolution using an
    efficient sub-pixel convolutional neural network.

    In Proceedings of the IEEE conference on computer vision
    and pattern recognition (pp. 1874-1883).

    https://www.cv-foundation.org/openaccess/content_cvpr_2016/
    papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Aitken, A., Ledig, C., Theis, L., Caballero, J., Wang, Z.
    and Shi, W., 2017.

    Checkerboard artifact free sub-pixel convolution: A note on
    sub-pixel convolution, resize convolution and convolution
    resize.

    arXiv preprint arXiv:1707.02937.

    https://arxiv.org/pdf/1707.02937.pdf
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Sugawara, Y., Shiota, S. and Kiya, H., 2018, October.

    Super-resolution using convolutional neural networks without
    any checkerboard artifacts.

    In 2018 25th IEEE International Conference on Image Processing
    (ICIP) (pp. 66-70). IEEE.

    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8451141
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    """

    def __init__(
        self,
        upsample_factor=2,
        n_output_chns=1,
        kernel_size=3,
        acti_func="tanh",
        feature_normalization=None,
        group_size=-1,
        with_bias=True,
        padding="REFLECT",
        use_icnr=False,
        use_avg=False,
        w_initializer=None,
        w_regularizer=None,
        b_initializer=None,
        b_regularizer=None,
        name="subpixel_cnn",
    ):
        """
        :param upsample_factor: zoom-factor/image magnification factor
        :param n_output_chns: the desired ammount of channels for the
        upsampled image
        :param kernel_size: the size of the convolutional kernels
        :param acti_func: activation function applied to first N - 1 layers
        :param feature_normalization: the type of feature normalization (e.g.
        batch, instance or group norm. Default None.
        :param group_size: size of the groups if groupnorm is chosen.
        :param with_bias: incorporate bias parameters in convolutional layers
        :param padding: padding applied in convolutional layers
        :param use_icnr: whether to use Aitken et al. initialization
        :param use_avg: whether to use Sugawara et al. post-processing
        """

        super(SubPixelLayer, self).__init__(name=name)

        if upsample_factor <= 0:
            raise ValueError("The upsampling factor must be strictly positive.")
        if int(upsample_factor) != float(upsample_factor) and use_icnr:
            raise ValueError(
                "If ICNR initialization is used the sample factor must be an integer"
            )
        if w_initializer is None and use_icnr:
            raise ValueError(
                "If ICNR initialization is used the weights initializer must be specified"
            )

        self.upsample_factor = upsample_factor
        self.kernel_size = kernel_size
        self.acti_func = acti_func
        self.use_avg = use_avg
        self.n_output_chns = n_output_chns

        self.conv_layer_params = {
            "with_bias": with_bias,
            "feature_normalization": feature_normalization,
            "group_size": group_size,
            "padding": padding,
            "w_initializer": w_initializer
            if not use_icnr
            else _ICNR(
                initializer=tf.keras.initializers.get(w_initializer),
                upsample_factor=upsample_factor,
            ),
            "b_initializer": b_initializer,
            "w_regularizer": w_regularizer,
            "b_regularizer": b_regularizer,
        }

    def layer_op(self, lr_image, is_training=True, keep_prob=1.0):
        input_shape = lr_image.get_shape().as_list()
        batch_size = input_shape.pop(0)

        if batch_size is None:
            raise ValueError("The batch size must be known and fixed.")
        if any(i is None or i <= 0 for i in input_shape):
            raise ValueError("The image shape must be known in advance.")

        # Making sure there are enough features channels for the
        # periodic shuffling
        features = ConvolutionalLayer(
            n_output_chns=(
                self.n_output_chns * self.upsample_factor ** (len(input_shape) - 1)
            ),
            kernel_size=self.kernel_size,
            acti_func=None,
            name="subpixel_conv",
            **self.conv_layer_params
        )(input_tensor=lr_image, is_training=is_training, keep_prob=keep_prob)

        # Setting the number of output features to the known value
        # obtained from the input shape results in a ValueError as
        # of TF 1.12
        sr_image = tf.contrib.periodic_resample.periodic_resample(
            values=features,
            shape=(
                [batch_size]
                + [self.upsample_factor * i for i in input_shape[:-1]]
                + [None]
            ),
            name="periodic_shuffle",
        )

        # Averaging out the values without downsampling to counteract
        # the periodicity of periodic shuffling as per Sugawara et al.
        if self.use_avg:
            sr_image = DownSampleLayer(
                func="AVG",
                kernel_size=self.upsample_factor,
                stride=1,
                padding="SAME",
                name="averaging",
            )(input_tensor=sr_image)

        return sr_image


class _ICNR:
    def __init__(self, initializer, upsample_factor=1):
        """
        :param initializer:  initializer used for sub kernels (orthogonal, glorot uniform, etc.)
        :param upsample_factor: upsample factor of sub pixel convolution
        """
        self.upsample_factor = upsample_factor
        self.initializer = initializer

    def __call__(self, shape, dtype, partition_info=None):
        shape = list(shape)
        if self.upsample_factor == 1:
            return self.initializer(shape)

        # Initializing W0 (enough kernels for one output channel)
        new_shape = shape[:-1] + [
            shape[-1] // (self.upsample_factor ** (len(shape) - 2))
        ]
        x = self.initializer(new_shape, dtype, partition_info)

        # Repeat the elements along the output dimension
        x = tf.keras.backend.repeat_elements(
            x=x, rep=self.upsample_factor ** (len(shape) - 2), axis=-1
        )

        return x
