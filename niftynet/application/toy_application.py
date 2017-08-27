from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import ApplicationNetFactory
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.sampler_random_vector import RandomVectorSampler
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.fully_connected import FullyConnectedLayer
from niftynet.network.base_net import BaseNet


class ToyApplication(BaseApplication):
    REQUIRED_CONFIG_SECTION = "TOY"

    def __init__(self, net_param, action_param, is_training):
        BaseApplication.__init__(self)
        tf.logging.info('starting toy application')
        self.is_training = is_training

        self.net_param = net_param
        self.action_param = action_param
        self.toy_param = None

    def initialise_dataset_loader(self, data_param=None, task_param=None):
        self.toy_param = task_param
        self.reader = ()

    def initialise_sampler(self):
        self.sampler = [
            RandomVectorSampler(
                names=('vectors',),
                vector_size=(self.toy_param.vector_size,),
                batch_size=self.net_param.batch_size,
                repeat=None)]

    def initialise_network(self):
        self.net = ApplicationNetFactory.create(self.net_param.name)()

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        with tf.name_scope('Optimiser'):
            optimiser_class = OptimiserFactory.create(
                name=self.action_param.optimiser)
            self.optimiser = optimiser_class.get_instance(
                learning_rate=self.action_param.lr)

        # a new pop_batch_op for each gpu tower
        data_x = self.get_sampler()[0].pop_batch_op()
        features = tf.cast(data_x['vectors'], tf.float32)
        features = tf.expand_dims(features, axis=-1)

        noise = tf.random_uniform(tf.shape(features), 0.0, 1.0)
        real_logits, fake_logits, fake_features = self.net(features, noise)

        batch_size = tf.shape(real_logits)[0]
        d_loss = \
            tf.losses.sparse_softmax_cross_entropy(
                tf.ones([batch_size, 1], tf.int32), real_logits) + \
            tf.losses.sparse_softmax_cross_entropy(
                tf.zeros([batch_size, 1], tf.int32), fake_logits)
        g_loss = \
            tf.losses.sparse_softmax_cross_entropy(
                tf.ones([batch_size, 1], tf.int32), fake_logits)

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

        outputs_collector.add_to_collection(
            var=d_loss, name='ave_d_loss', average_over_devices=True,
            collection=CONSOLE)
        outputs_collector.add_to_collection(
            var=g_loss, name='ave_g_loss', average_over_devices=True,
            collection=CONSOLE)

        outputs_collector.add_to_collection(
            var=d_loss, name='d_loss', average_over_devices=False,
            collection=TF_SUMMARIES)
        outputs_collector.add_to_collection(
            var=g_loss, name='g_loss', average_over_devices=False,
            collection=TF_SUMMARIES)

        g_mean, g_var = tf.nn.moments(fake_features, axes=[0, 1, 2])
        outputs_collector.add_to_collection(
            var=g_mean, name='generated_mean', average_over_devices=True,
            collection=TF_SUMMARIES)
        outputs_collector.add_to_collection(
            var=g_var, name='generated_variance', average_over_devices=True,
            collection=TF_SUMMARIES)

        outputs_collector.add_to_collection(
            var=features, name='original_distribution',
            average_over_devices=False,
            collection=TF_SUMMARIES, summary_type='histogram')
        outputs_collector.add_to_collection(
            var=fake_features, name='generated_distribution',
            average_over_devices=False,
            collection=TF_SUMMARIES, summary_type='histogram')


    def interpret_output(self, batch_output):
        return True


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
        batch_size = features.get_shape().as_list()[0]
        conv_1 = ConvolutionalLayer(10, 3, with_bn=False)
        fc_1 = FullyConnectedLayer(2, with_bn=False)

        hidden_feature = conv_1(features, is_training=True)
        hidden_feature = tf.reshape(hidden_feature, [batch_size, -1])
        logits = fc_1(hidden_feature, is_training=True)
        return logits


class GNet(BaseNet):
    def __init__(self):
        BaseNet.__init__(self, name='G')

    def layer_op(self, noise):
        n_chns = noise.get_shape()[-1]
        conv_1 = ConvolutionalLayer(
            20, 10, with_bn=False,
            w_initializer=tf.random_uniform_initializer(-5.0, 5.0))
        conv_2 = ConvolutionalLayer(
            n_chns, 10, with_bn=False, with_bias=True,
            w_initializer=tf.random_uniform_initializer(-1.0, 1.0),
            b_initializer=tf.random_uniform_initializer(4.0, 5.0))
        hidden_feature = conv_1(noise, is_training=True)
        fake_features = conv_2(hidden_feature, is_training=True)
        return fake_features

