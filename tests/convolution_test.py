from __future__ import division, absolute_import, print_function

import functools as ft
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from niftynet.layer.convolution import ConvLayer
from niftynet.layer.convolution import ConvolutionalLayer
from tests.niftynet_testcase import NiftyNetTestCase

class ConvTest(NiftyNetTestCase):
    def get_3d_input(self):
        input_shape = (2, 16, 16, 16, 8)
        x_3d = tf.ones(input_shape)
        return x_3d

    def get_2d_input(self):
        input_shape = (2, 16, 16, 8)
        x_2d = tf.ones(input_shape)
        return x_2d

    def _test_conv_output_shape(self,
                                rank,
                                param_dict,
                                output_shape):
        if rank == 2:
            input_data = self.get_2d_input()
        elif rank == 3:
            input_data = self.get_3d_input()

        conv_layer = ConvLayer(**param_dict)
        output_data = conv_layer(input_data)
        print(conv_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_value = sess.run(output_data)
            self.assertAllClose(output_shape, output_value.shape)

    def _test_conv_layer_output_shape(self,
                                      rank,
                                      param_dict,
                                      output_shape,
                                      is_training=None,
                                      dropout_prob=None):
        if rank == 2:
            input_data = self.get_2d_input()
        elif rank == 3:
            input_data = self.get_3d_input()

        conv_layer = ConvolutionalLayer(**param_dict)
        output_data = conv_layer(input_data,
                                 is_training=is_training,
                                 keep_prob=dropout_prob)
        print(conv_layer)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            output_value = sess.run(output_data)
            self.assertAllClose(output_shape, output_value.shape)

    def _test_extended_conv(self, orig_input, init_dict):
        """
        Tests the extended padding options of ConvLayer
        """

        def _w_init(shape, dtype=tf.float32, **kwargs):
            data = np.arange(ft.reduce(lambda prod, x: prod*x, shape, 1))\
                     .astype(np.float32)
            data *= 2.374/data.mean()
            data -= data.mean()

            return tf.constant(data.reshape(shape), dtype=dtype)

        def _b_init(shape, dtype=tf.float32, **kwargs):
            data = np.arange(shape[0]).astype(np.float32)
            data *= 0.273/data.mean()
            data -= data.mean()

            return tf.constant(data.reshape(shape), dtype=dtype)

        init_dict['w_initializer'] = _w_init
        init_dict['b_initializer'] = _b_init

        conv_layer = ConvLayer(**init_dict)
        small_output = conv_layer(tf.constant(orig_input))

        input_shape = orig_input.shape
        multiplier = init_dict['kernel_size'] + init_dict['dilation'] \
            + init_dict['stride']
        pad = [d*multiplier for d in input_shape[1:-1]]
        paddings = [(0, 0)] + [(p, p) for p in pad] + [(0, 0)]

        if init_dict['padding'] == 'CONSTANT':
            opts = {'constant_values': init_dict.get('padding_constant', 0)}
        else:
            opts = {}

        enlarged_input = np.pad(orig_input,
                                paddings,
                                init_dict['padding'].lower(),
                                **opts)

        conv_layer.padding = 'SAME'
        large_output = conv_layer(tf.constant(enlarged_input))

        def _extract_valid_region(output_tensor, target_tensor):
            output_shape = output_tensor.shape
            target_shape = target_tensor.shape
            extr_slices = []
            for d in range(len(target_shape)):
                opad = (output_shape[d] - target_shape[d])//2
                extr_slices.append(slice(
                    opad, opad + target_shape[d]))

            return output_tensor[tuple(extr_slices)]

        assert np.square(
            _extract_valid_region(enlarged_input, orig_input) - orig_input).sum() \
            <= 1e-6*np.square(orig_input).sum()

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())

            small_value = sess.run(small_output)
            large_value = sess.run(large_output)

            extr_value = _extract_valid_region(large_value, small_value)

            print(np.square(small_value - extr_value).sum()/np.square(extr_value).sum())

            self.assertAllClose(small_value, extr_value, rtol=1e-3)

    def _get_pad_test_input_3d(self):
        data = np.arange(1024, dtype=np.float32)

        return data.reshape([1, 16, 4, 4, 4])

    def _get_pad_test_input_2d(self):
        data = np.arange(256, dtype=np.float32)

        return data.reshape([4, 8, 4, 2])

    # padding tests
    def _test_extended_padding(self, pad, do_2d):
        batch = self._get_pad_test_input_2d() if do_2d \
            else self._get_pad_test_input_3d()

        const = 127.23
        min_dim = min(batch.shape[1:-1]) - 1
        for ks in (2, min_dim):
            for ds in (1, min_dim):
                name = 'pad_test_conv' + ('2' if do_2d else '3')
                name += "%i_%i" % (ks, ds)
                init_dict = {'n_output_chns': 4,
                             'kernel_size': ks,
                             'stride': 1,
                             'dilation': ds,
                             'padding': pad,
                             'name': name}

                if ds%2 == 0:
                    init_dict['padding_constant'] = const

                self._test_extended_conv(batch, init_dict)

    def test_2d_const_padding(self):
        self._test_extended_padding('CONSTANT', True)

    def test_2d_reflect_padding(self):
        self._test_extended_padding('REFLECT', True)

    def test_2d_symmetric_padding(self):
        self._test_extended_padding('SYMMETRIC', True)

    def test_3d_const_padding(self):
        self._test_extended_padding('CONSTANT', False)

    def test_3d_reflect_padding(self):
        self._test_extended_padding('REFLECT', False)

    def test_3d_symmetric_padding(self):
        self._test_extended_padding('SYMMETRIC', False)

    # 3d tests
    def test_3d_conv_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 16, 10))

    def test_3d_conv_full_kernel_size(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 3, 1],
                       'stride': 1}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 16, 10))

    def test_3d_conv_full_strides(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 3, 1],
                       'stride': [1, 1, 2]}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 8, 10))

    def test_3d_anisotropic_conv(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 2, 1],
                       'stride': [1, 1, 2]}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 8, 10))

    def test_3d_conv_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 1, 3],
                       'stride': [1, 1, 2],
                       'with_bias': True}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 8, 10))

    def test_conv_3d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 1, 3],
                       'stride': [2, 2, 2],
                       'with_bias': True,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 8, 8, 8, 10))

    def test_3d_convlayer_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 16, 10))

    def test_3d_convlayer_dilation_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1,
                       'dilation': [1, 2, 1]}
        self._test_conv_output_shape(rank=3,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 16, 10))

    def test_3d_convlayer_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1,
                       'with_bias': True,
                       'feature_normalization': None}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 16, 16, 10))

    def test_convlayer_3d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1,
                       'with_bias': True,
                       'feature_normalization': None,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 16, 16, 10))

    def test_convlayer_3d_bn_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 1, 2],
                       'stride': 1,
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 16, 16, 10),
                                           is_training=True)

    def test_convlayer_3d_bn_reg_prelu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 1, 2],
                       'stride': [1, 1, 2],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 16, 8, 10),
                                           is_training=True)

    def test_convlayer_3d_relu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 1, 2],
                       'stride': [1, 2, 2],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'relu',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 8, 8, 10),
                                           is_training=True)

    def test_convlayer_3d_bn_reg_dropout_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 1, 2],
                       'stride': [1, 2, 2],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu'}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 8, 8, 10),
                                           is_training=True,
                                           dropout_prob=0.4)

    def test_convlayer_3d_bn_reg_dropout_valid_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 3, 2],
                       'stride': [2, 2, 3],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'acti_func': 'prelu',
                       'padding': 'VALID'}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 6, 7, 5, 10),
                                           is_training=True,
                                           dropout_prob=0.4)

    def test_convlayer_3d_group_reg_dropout_valid_shape(self):
        input_param = {'n_output_chns': 8,
                       'kernel_size': [5, 3, 2],
                       'stride': [2, 2, 3],
                       'with_bias': False,
                       'feature_normalization': 'group',
                       'group_size': 4,
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=3,
                                           param_dict=input_param,
                                           output_shape=(2, 8, 8, 6, 8),
                                           is_training=True,
                                           dropout_prob=0.4)

    # 2d tests
    def test_2d_conv_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 3],
                       'stride': [2, 2]}
        self._test_conv_output_shape(rank=2,
                                     param_dict=input_param,
                                     output_shape=(2, 8, 8, 10))

    def test_2d_conv_dilation_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 3],
                       'stride': [1, 1],
                       'dilation': [2, 1]}
        self._test_conv_output_shape(rank=2,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 10))

    def test_2d_conv_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [5, 3],
                       'stride': [1, 2],
                       'with_bias': True}
        self._test_conv_output_shape(rank=2,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 8, 10))

    def test_conv_2d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 5,
                       'stride': 1,
                       'with_bias': True,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_output_shape(rank=2,
                                     param_dict=input_param,
                                     output_shape=(2, 16, 16, 10))

    def test_2d_convlayer_default_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 2,
                       'stride': 1,
                       'with_bias': True}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 16, 10),
                                           is_training=True)

    def test_2d_convlayer_bias_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 2,
                       'stride': [2, 1],
                       'with_bias': True,
                       'feature_normalization': None}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 8, 16, 10))

    def test_convlayer_2d_bias_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 5],
                       'stride': [2, 1],
                       'with_bias': True,
                       'feature_normalization': None,
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 8, 16, 10))

    def test_convlayer_2d_bn_reg_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 5],
                       'stride': [2, 1],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'w_regularizer': regularizers.l2_regularizer(0.5),
                       'b_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 8, 16, 10),
                                           is_training=True)

    def test_convlayer_2d_bn_reg_prelu_2_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': [2, 1],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu'}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 8, 16, 10),
                                           is_training=True)

    def test_convlayer_2d_relu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': [3, 1],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'relu'}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 6, 16, 10),
                                           is_training=True)

    def test_convlayer_2d_bn_reg_prelu_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': 3,
                       'stride': 1,
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 16, 16, 10),
                                           is_training=True)

    def test_convlayer_2d_bn_reg_valid_shape(self):
        input_param = {'n_output_chns': 10,
                       'kernel_size': [3, 2],
                       'stride': [2, 3],
                       'with_bias': False,
                       'feature_normalization': 'batch',
                       'acti_func': 'prelu',
                       'padding': 'VALID',
                       'w_regularizer': regularizers.l2_regularizer(0.5)}
        self._test_conv_layer_output_shape(rank=2,
                                           param_dict=input_param,
                                           output_shape=(2, 7, 5, 10),
                                           is_training=True)


if __name__ == "__main__":
    tf.test.main()
