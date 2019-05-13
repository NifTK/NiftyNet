import operator

import tensorflow as tf

import niftynet.layer.activation as nnla
import niftynet.layer.convolution as nnlc
import niftynet.layer.deconvolution as nnldc
import niftynet.layer.fully_connected as nnlfc
import niftynet.layer.gaussian_sampler as nnlgs
import niftynet.layer.gn as nnlg
import niftynet.network.base_net as nnnbn

import functools as ft


class GuuNet(nnnbn.BaseNet):

    def __init__(
            self, num_classes=1,
            w_initializer=None, w_regularizer=None,
            b_initializer=None, b_regularizer=None,
            acti_func=None,
            has_seg_feature=True, has_seg_logvar_decoder=True,
            has_autoencoder_feature=True, has_autoencoder_logvar_decoder=True,
            name='GuuNet'):
        super(GuuNet, self).__init__(name=name)

        self.num_classes = num_classes
        self.w_initializer = w_initializer
        self.w_regularizer = w_regularizer
        self.b_initializer = b_initializer
        self.b_regularizer = b_regularizer
        self.has_seg_feature = has_seg_feature
        self.has_seg_logvar_decoder = has_seg_logvar_decoder
        self.has_autoencoder_feature = has_autoencoder_feature
        self.has_autoencoder_logvar_decoder = has_autoencoder_logvar_decoder
        self.gaussian_segmentation = True

    @staticmethod
    def encoder_block(_, channels, encoder_gn, encoder_acti, encoder_conv, i, j, skips):
        l = _
        _ = nnlg.GNLayer(name='gn_{}_{}_{}'.format(i, j, 0), **encoder_gn)(_)
        _ = nnla.ActiLayer(encoder_acti, name='acti_{}_{}_{}'.format(i, j, 0))(_)
        _ = nnlc.ConvLayer(channels << i, name='conv_{}_{}_{}'.format(i, j, 0), **encoder_conv)(_)
        _ = nnlg.GNLayer(name='gn_{}_{}_{}'.format(i, j, 1), **encoder_gn)(_)
        _ = nnla.ActiLayer(encoder_acti, name='acti_{}_{}_{}'.format(i, j, 1))(_)
        _ = nnlc.ConvLayer(channels << i, name='conv_{}_{}_{}'.format(i, j, 1), **encoder_conv)(_)
        _ = tf.add(l, _, name='add_{}_{}'.format(i, j))
        if skips is not None:
            skips.append(_)
        return _

    @staticmethod
    def decoder_block(_, channels, encoder_gn, encoder_acti, encoder_conv, i, j):
        return GuuNet.encoder_block(
            _, channels, encoder_gn, encoder_acti, encoder_conv, i, j, None)

    @staticmethod
    def encoder_down(_, channels, down_conv, i):
        _ = nnlc.ConvLayer(
            channels << (i + 1), name='down_conv_{}'.format(i), **down_conv)(_)
        return _

    @staticmethod
    def decoder_up(_, channels, up_conv, up_upsample, i, skips = None):
        _ = nnlc.ConvLayer(
            channels << (i + 1), name='up_conv_{}'.format(i), **up_conv)(_)
        _ = nnldc.DeconvLayer(
            channels << i, name='up_upsample_{}'.format(i), **up_upsample)(_)
        if skips is not None:
            _ = tf.add(_, skips[i], name='up_encoder_add_{}'.format(i))
        return _

    @staticmethod
    def seg_decoder_up(_, channels, up_conv, up_upsample, i, skips):
        return GuuNet.decoder_up(_, channels, up_conv, up_upsample, i, skips)

    @staticmethod
    def vae_decoder_up(_, channels, up_conv, up_upsample, i):
        return GuuNet.decoder_up(_, channels, up_conv, up_upsample, i, None)

    @staticmethod
    def dense_down(_, conv_channels, latent_channels, down_gn, down_acti, down_conv, down_fc):
        _ = nnlg.GNLayer(name='down_gn', **down_gn)(_)
        _ = nnla.ActiLayer(down_acti, name='down_acti')(_)
        _ = nnlc.ConvLayer(conv_channels, name='down_conv', **down_conv)(_)
        restore_shape = _.shape
        _ = tf.reshape(_, [-1, ft.reduce(operator.mul, _.shape[1:], 1)])
        _ = nnlfc.FullyConnectedLayer(latent_channels, name='down_fully_conn', **down_fc)(_)
        return _, restore_shape

    @staticmethod
    def dense_up(_, conv_channels, restore_shape, up_fc, up_acti, up_conv, up_upsample):
        _ = nnlfc.FullyConnectedLayer(
            ft.reduce(operator.mul, restore_shape[1:], 1),
            name='up_fully_conn', **up_fc)(_)
        _ = tf.reshape(_, [-1] + restore_shape[1:].as_list())
        _ = nnla.ActiLayer(up_acti, name='up_acti')(_)
        _ = nnlc.ConvLayer(conv_channels, name='up_conv', **up_conv)(_)
        _ = nnldc.DeconvLayer(conv_channels, name='up_upsample', **up_upsample)(_)
        return _

    def layer_op(self, images, is_training):
        outputs = dict()
        internal = list()

        bc = 32  # base_channels
        fsc = 3  # final channels
        fvc = 2
        gsc = 16  # gaussian sampler down convolution channels
        lc = 256  # latent channels
        layers = 3
        init_conv = {'kernel_size': 3, 'stride': 1}
        encoder_gn = {'group_size': 8, 'regularizer': None, 'eps': 1e-5}
        encoder_acti = 'relu'
        encoder_conv = {'kernel_size': 3, 'stride': 1}
        down_conv = {'kernel_size': 3, 'stride': 2}
        up_conv = {'kernel_size': 1, 'stride': 1}
        up_upsample = {'kernel_size': 2, 'stride': 2}
        final_conv = {'kernel_size': 1, 'stride': 1}
        dense_down_gn = {'group_size': 8, 'regularizer': None, 'eps': 1e-5}
        dense_down_acti = 'relu'
        dense_down_conv = {'kernel_size': 3, 'stride': 2}
        dense_down_fc = {'with_bias': False, 'with_bn': False}
        dense_up_fc = {'with_bias': False, 'with_bn': False}
        dense_up_conv = {'kernel_size': 1, 'stride': 1}
        dense_up_acti = 'relu'
        dense_up_uplinear = {'kernel_size': 2, 'stride': 2}

        repeats = [1, 2, 2, 4]
        up_repeats = [2, 2, 2]

        outputs = dict()
        skips = list()
        _ = images
        with tf.variable_scope('encoder'):
            _ = nnlc.ConvLayer(bc, name='init_conv', **init_conv)(_)

            for i in range(0, layers+1):
                for j in range(0, repeats[i]):
                    _ = self.encoder_block(
                        _, bc, encoder_gn, encoder_acti, encoder_conv, i, j,
                        skips if repeats[i] - j == 1 else None)
                if i < layers:
                    _ = self.encoder_down(_, bc, down_conv, i)

        with tf.variable_scope('gaussian_sampler'):
            _, restore_shape = self.dense_down(
                _, gsc, lc, dense_down_gn, dense_down_acti, dense_down_conv,
                dense_down_fc)

            _, pmeans, plogvars = nnlgs.GaussianSampler(
                lc / 2, 1, 10, -10, name='gaussian_sampler')(_, is_training)

            gaussian_sampler_out = self.dense_up(
                _, bc << layers, restore_shape, dense_up_fc, dense_up_acti,
                dense_up_conv, dense_up_uplinear)

        if self.has_seg_feature:
            with tf.variable_scope('seg_mean_decoder'):
                if self.gaussian_segmentation is True:
                    _ = gaussian_sampler_out
                else:
                    _ = skips[-1]
                for i in reversed(range(0, layers)):
                    _ = self.decoder_up(_, bc, up_conv, up_upsample, i, skips)
                    _ = self.decoder_block(
                        _, bc, encoder_gn, encoder_acti, encoder_conv, i, 0)

                seg_means_out = nnlc.ConvLayer(fsc, name='final_seg_conv', **final_conv)(_)

            if self.has_seg_logvar_decoder:
                with tf.variable_scope('seg_logvar_decoder'):
                    if self.gaussian_segmentation is True:
                        _ = gaussian_sampler_out
                    else:
                        _ = skips[-1]
                    for i in reversed(range(0, layers)):
                        _ = self.decoder_up(_, bc, up_conv, up_upsample, i, skips)
                        _ = self.decoder_block(
                            _, bc, encoder_gn, encoder_acti, encoder_conv, i, 0)

                    seg_logvars_out = nnlc.ConvLayer(fsc, name='final_seg_conv', **final_conv)(_)

        if self.has_autoencoder_feature:
            with tf.variable_scope('vae_mean_decoder'):
                _ = gaussian_sampler_out
                for i in reversed(range(0, layers)):
                    _ = self.vae_decoder_up(_, bc, up_conv, up_upsample, i)
                    for j in range(0, up_repeats[i]):
                        _ = self.decoder_block(
                            _, bc, encoder_gn, encoder_acti, encoder_conv, i, j)

                img_means_out = nnlc.ConvLayer(fvc, name='final_img_conv', **final_conv)(_)

            if self.has_autoencoder_logvar_decoder:
                with tf.variable_scope('vae_logvar_decoder'):
                    _ = gaussian_sampler_out
                    for i in reversed(range(0, layers)):
                        _ = self.vae_decoder_up(_, bc, up_conv, up_upsample, i)
                        for j in range(0, up_repeats[i]):
                            _ = self.decoder_block(
                                _, bc, encoder_gn, encoder_acti, encoder_conv, i, j)

                    img_logvars_out = nnlc.ConvLayer(fvc, name='final_img_conv', **final_conv)(_)

        if self.has_seg_feature:
            outputs['seg_means'] = seg_means_out
            if self.has_seg_logvar_decoder:
                outputs['seg_logvars'] = seg_logvars_out

        if self.has_autoencoder_feature:
            outputs['image_means'] = img_means_out
            if self.has_autoencoder_logvar_decoder:
                outputs['image_logvars'] = img_logvars_out

            outputs['posterior_means'] = pmeans
            outputs['posterior_logvars'] = plogvars

        return outputs, dict()
