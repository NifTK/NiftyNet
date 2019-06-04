# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import ApplicationNetFactory
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.sampler_random_vector_v2 import RandomVectorSampler
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.network.base_net import BaseNet

class ToyApplication(BaseApplication):
    REQUIRED_CONFIG_SECTION = "TOY"

    def __init__(self, net_param, action_param, action):
        BaseApplication.__init__(self)
        tf.logging.info('starting toy application')
        self.action = action

        self.net_param = net_param
        self.action_param = action_param
        self.toy_param = None

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        self.toy_param = task_param
        self.readers = []

    def initialise_sampler(self):
        self.sampler = [
            [RandomVectorSampler(
                names=('vectors',),
                vector_size=(self.toy_param.vector_size,),
                batch_size=self.net_param.batch_size,
                repeat=None,
                mean=self.toy_param.mean,
                stddev=self.toy_param.stddev)]]

    def initialise_network(self):
        self.net = ApplicationNetFactory.create(self.net_param.name)()

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        print(vars(self.action_param))
        self.patience = self.action_param.patience
        with tf.name_scope('Optimiser'):
            optimiser_class = OptimiserFactory.create(
                name=self.action_param.optimiser)
            self.optimiser = optimiser_class.get_instance(
                learning_rate=self.action_param.lr)

        fake_features, fake_logits, real_logits = self.feed_forward()

        d_loss, g_loss = self.compute_loss(fake_logits, real_logits)

        with tf.name_scope('ComputeGradients'):
            d_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=self.net.d_net.layer_scope().name)
            g_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=self.net.g_net.layer_scope().name)
            grads_d = self.optimiser.compute_gradients(
                d_loss, var_list=d_vars)
            grads_g = self.optimiser.compute_gradients(
                g_loss, var_list=g_vars)
            grads = [grads_d, grads_g]
            gradients_collector.add_to_collection(grads)

        self.collect_results(d_loss, fake_features, g_loss, outputs_collector)

    def collect_results(self, d_loss, fake_features, g_loss, outputs_collector):
        outputs_collector.add_to_collection(
            var=d_loss, name='d_loss', average_over_devices=True,
            collection=TF_SUMMARIES)
        outputs_collector.add_to_collection(
            var=g_loss, name='g_loss', average_over_devices=True,
            collection=TF_SUMMARIES)
        g_mean, g_var = tf.nn.moments(fake_features, axes=[0, 1, 2])
        g_var = tf.sqrt(g_var)
        outputs_collector.add_to_collection(
            var=g_mean, name='mean', average_over_devices=True,
            collection=CONSOLE)
        outputs_collector.add_to_collection(
            var=g_var, name='var', average_over_devices=True,
            collection=CONSOLE)
        outputs_collector.add_to_collection(
            var=g_loss, name='total_loss', average_over_devices=True,
            collection=CONSOLE)
        outputs_collector.add_to_collection(
            var=g_mean, name='generated_mean', average_over_devices=False,
            collection=TF_SUMMARIES)
        outputs_collector.add_to_collection(
            var=g_var, name='generated_variance', average_over_devices=False,
            collection=TF_SUMMARIES)

    def compute_loss(self, fake_logits, real_logits):
        d_loss = tf.reduce_mean(real_logits - fake_logits)
        g_loss = tf.reduce_mean(fake_logits)
        return d_loss, g_loss

    def feed_forward(self):
        # a new pop_batch_op for each gpu tower
        data_x = self.get_sampler()[0][0].pop_batch_op()
        features = tf.cast(data_x['vectors'], tf.float32, name='sampler_input')
        features = tf.expand_dims(features, axis=-1, name='feature_input')
        noise = tf.random_uniform(tf.shape(features), 0.0, 1.0)
        real_logits, fake_logits, fake_features = self.net(features, noise)
        return fake_features, fake_logits, real_logits

    def interpret_output(self, batch_output):
        return True


class ToyApplicationMultOpti(ToyApplication):

    def __init__(self, net_param, action_param, action):
        ToyApplication.__init__(self, net_param, action_param, action)
        tf.logging.info('starting toy application using multiple optimiser')

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        self.patience = self.action_param.patience
        print(vars(self.action_param))
        self.optimiser = dict()
        with tf.name_scope('OptimiserGen'):
            optimiser_class = OptimiserFactory.create(
                name=self.action_param.optimiser)
            self.optimiser['gen'] = optimiser_class.get_instance(
                learning_rate=self.action_param.lr)

        # 2nd optimiser could be initialized different
        with tf.name_scope('OptimiserDis'):
            optimiser_class = OptimiserFactory.create(
                name=self.action_param.optimiser)
            self.optimiser['dis'] = optimiser_class.get_instance(
                learning_rate=self.action_param.lr)

        fake_features, fake_logits, real_logits = self.feed_forward()

        d_loss, g_loss = self.compute_loss(fake_logits, real_logits)

        grads = dict()
        with tf.name_scope('ComputeGradientsD'):
            d_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=self.net.d_net.layer_scope().name)
            grads['dis'] = self.optimiser['dis'].compute_gradients(
                d_loss, var_list=d_vars)
        with tf.name_scope('ComputeGradientsG'):
            g_vars = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=self.net.g_net.layer_scope().name)
            grads['gen'] = self.optimiser['gen'].compute_gradients(
                g_loss, var_list=g_vars)

        gradients_collector.add_to_collection(grads)

        self.collect_results(d_loss, fake_features, g_loss, outputs_collector)


class TinyNet(BaseNet):
    def __init__(self):
        BaseNet.__init__(self, name='tinynet')
        self.d_net = DNet()
        self.g_net = GNet()

    def layer_op(self, features, noise):
        fake_features = self.g_net(noise)

        real_logits = self.d_net(features)
        fake_logits = self.d_net(fake_features)
        return real_logits, fake_logits, fake_features


class DNet(BaseNet):
    def __init__(self):
        BaseNet.__init__(self, name='D')

    def layer_op(self, features):
        batch_size = features.shape.as_list()[0]
        conv_1 = ConvolutionalLayer(
            20, 3, feature_normalization=None, with_bias=True, acti_func='relu')
        fc_1 = FullyConnectedLayer(
            20, feature_normalization=None, with_bias=True, acti_func='relu')
        fc_2 = FullyConnectedLayer(
            2, feature_normalization=None, with_bias=True)

        hidden_feature = conv_1(features, is_training=True)
        hidden_feature = tf.reshape(hidden_feature, [batch_size, -1])
        hidden_feature = fc_1(hidden_feature, is_training=True)
        logits = fc_2(hidden_feature, is_training=True)
        return logits


class GNet(BaseNet):
    def __init__(self):
        BaseNet.__init__(self, name='G')

    def layer_op(self, noise):
        n_chns = noise.shape[-1]
        conv_1 = ConvolutionalLayer(
            20, 10, feature_normalization='batch', acti_func='selu', with_bias=True)
        conv_2 = ConvolutionalLayer(
            20, 10, feature_normalization='batch', acti_func='selu', with_bias=True)
        conv_3 = ConvolutionalLayer(
            n_chns, 10, feature_normalization=None, with_bias=True)
        hidden_feature = conv_1(noise, is_training=True)
        hidden_feature = conv_2(hidden_feature, is_training=True)
        fake_features = conv_3(hidden_feature, is_training=True)
        return fake_features
