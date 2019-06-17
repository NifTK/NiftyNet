from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

from niftynet.layer.channel_sparse_convolution import ChannelSparseConvolutionalLayer
from tests.niftynet_testcase import NiftyNetTestCase

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2

def get_config():
    rewrite_options = rewriter_config_pb2.RewriterConfig(
        layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF)
    graph_options = config_pb2.GraphOptions(
        rewrite_options=rewrite_options, build_cost_model=1)
    config = config_pb2.ConfigProto(graph_options=graph_options)
    return config


class ChannelSparseConvolutionalLayerTest(NiftyNetTestCase):
    def test_3d_shape(self):
        x = tf.random_normal(shape=[2,4,5,6,4])
        conv1 = ChannelSparseConvolutionalLayer(4)
        conv2 = ChannelSparseConvolutionalLayer(8, kernel_size=[1,1,3])
        conv3 = ChannelSparseConvolutionalLayer(4, acti_func='relu')
        conv4 = ChannelSparseConvolutionalLayer(8, feature_normalization=None)
        conv5 = ChannelSparseConvolutionalLayer(4, with_bias=True)
        x1, mask1=conv1(x, None, True, 1.)
        x2, mask2=conv2(x1, mask1, True, 1.)
        x3, mask3=conv3(x2, mask2, True, .5)
        x4, mask4=conv4(x3, mask3, True, .75)
        x5, mask5=conv5(x4, mask4, True, 1.)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out1, out2, out3, out4, out5 = sess.run([x1,x2,x3,x4,x5])
        self.assertAllClose([2,4,5,6,4], out1.shape)
        self.assertAllClose([2,4,5,6,8], out2.shape)
        self.assertAllClose([2,4,5,6,2], out3.shape)
        self.assertAllClose([2,4,5,6,6], out4.shape)
        self.assertAllClose([2,4,5,6,4], out5.shape)

    def test_2d_shape(self):
        x = tf.random_normal(shape=[2,4,5,4])
        conv1 = ChannelSparseConvolutionalLayer(4)
        conv2 = ChannelSparseConvolutionalLayer(8, kernel_size=[1,1,3])
        conv3 = ChannelSparseConvolutionalLayer(4, acti_func='relu')
        conv4 = ChannelSparseConvolutionalLayer(8, feature_normalization=None)
        conv5 = ChannelSparseConvolutionalLayer(4, with_bias=True)
        x1, mask1=conv1(x, None, True, 1.)
        x2, mask2=conv2(x1, mask1, True, 1.)
        x3, mask3=conv3(x2, mask2, True, .5)
        x4, mask4=conv4(x3, mask3, True, .75)
        x5, mask5=conv5(x4, mask4, True, 1.)

        with self.cached_session(config=get_config()) as sess:
            sess.run(tf.global_variables_initializer())
            out1, out2, out3, out4, out5 = sess.run([x1,x2,x3,x4,x5])
        self.assertAllClose([2,4,5,4], out1.shape)
        self.assertAllClose([2,4,5,8], out2.shape)
        self.assertAllClose([2,4,5,2], out3.shape)
        self.assertAllClose([2,4,5,6], out4.shape)
        self.assertAllClose([2,4,5,4], out5.shape)

    def test_masks(self):
        x = tf.random_normal(shape=[2,4,5,4])
        conv1 = ChannelSparseConvolutionalLayer(10)
        conv2 = ChannelSparseConvolutionalLayer(10)
        conv3 = ChannelSparseConvolutionalLayer(10)
        conv4 = ChannelSparseConvolutionalLayer(10)
        x1, mask1=conv1(x, None, True, 1.)
        x2, mask2=conv2(x1, mask1, True, .5)
        x3, mask3=conv3(x2, mask2, True, .2)
        x4, mask4=conv4(x3, mask3, True, 1.)

        with self.cached_session(config=get_config()) as sess:
            sess.run(tf.global_variables_initializer())
            out1, out2, out3, out4 = sess.run([mask1, mask2, mask3, mask4])
        self.assertAllClose([10, 5, 2, 10], [np.sum(out1),
                                             np.sum(out2),
                                             np.sum(out3),
                                             np.sum(out4)])

if __name__ == "__main__":
    tf.test.main()
