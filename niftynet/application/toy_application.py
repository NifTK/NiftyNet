from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import ApplicationNetFactory
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.sampler_random_vector import RandomVectorSampler
from niftynet.layer.convolution import ConvolutionalLayer
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
        output = self.net(features)
        targets = tf.cast(features, tf.float32)
        loss = tf.reduce_mean(tf.square(output - targets))

        # variables to display
        outputs_collector.add_to_collection(
            var=loss, name='loss', average_over_devices=False,
            collection=CONSOLE)
        outputs_collector.add_to_collection(
            var=loss, name='ave_loss', average_over_devices=True,
            collection=CONSOLE)
        outputs_collector.add_to_collection(
            var=loss, name='loss', average_over_devices=False,
            collection=TF_SUMMARIES)
        outputs_collector.add_to_collection(
            var=loss, name='ave_loss', average_over_devices=True,
            collection=TF_SUMMARIES)

        with tf.name_scope('ComputeGradients'):
            grads = self.optimiser.compute_gradients(loss)
            gradients_collector.add_to_collection(grads)

    def interpret_output(self, batch_output):
        return True

class TinyNet(BaseNet):
    def __init__(self):
        BaseNet.__init__(self)

    def layer_op(self, features):
        conv_1 = ConvolutionalLayer(10, 3)
        conv_2 = ConvolutionalLayer(10, 3)
        conv_3 = ConvolutionalLayer(1, 3)
        hidden_feature = conv_1(features, is_training=True)
        hidden_feature = conv_2(hidden_feature, is_training=True)
        output = conv_3(hidden_feature, is_training=True)
        return output
