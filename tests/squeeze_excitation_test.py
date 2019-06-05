# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from niftynet.layer.squeeze_excitation import ChannelSELayer
from niftynet.layer.squeeze_excitation import SpatialSELayer
from niftynet.layer.squeeze_excitation import ChannelSpatialSELayer
from tests.niftynet_testcase import NiftyNetTestCase

class SETest(NiftyNetTestCase):
    def test_cSE_3d_shape(self):
        input_shape = (2, 16, 16, 16, 32)
        x = tf.ones(input_shape)
        se_layer = ChannelSELayer()
        out_se = se_layer(x)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_se)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            
    def test_sSE_3d_shape(self):
        input_shape = (2, 16, 16, 16, 32)
        x = tf.ones(input_shape)
        se_layer = SpatialSELayer()
        out_se = se_layer(x)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_se)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            
    def test_csSE_3d_shape(self):
        input_shape = (2, 16, 16, 16, 32)
        x = tf.ones(input_shape)
        se_layer = ChannelSpatialSELayer()
        out_se = se_layer(x)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_se)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)

    def test_cSE_2d_shape(self):
        input_shape = (2, 16, 16, 32)
        x = tf.ones(input_shape)
        se_layer = ChannelSELayer()
        out_se = se_layer(x)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_se)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            
    def test_sSE_2d_shape(self):
        input_shape = (2, 16, 16, 32)
        x = tf.ones(input_shape)
        se_layer = SpatialSELayer()
        out_se = se_layer(x)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_se)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)
            
    def test_csSE_2d_shape(self):
        input_shape = (2, 16, 16, 32)
        x = tf.ones(input_shape)
        se_layer = ChannelSpatialSELayer()
        out_se = se_layer(x)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_se)
            x_shape = tuple(x.shape.as_list())
            self.assertAllClose(x_shape, out.shape)

    def test_cSE_3d_excitation_op(self):
        input_shape = (2, 16, 16, 16, 32)
        x = tf.random_uniform(input_shape,seed=0)
        se_layer = ChannelSELayer()
        out_se = se_layer(x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x=sess.run(x)
            x_0_0=float(x[0,0,0,0,0])
            x_1_0=float(x[0,1,0,0,0])
            x_0_1=float(x[0,0,0,0,1])
            x_1_1=float(x[0,1,0,0,1])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_se)
            out_0_0=float(out[0,0,0,0,0])
            out_1_0=float(out[0,1,0,0,0])
            out_0_1=float(out[0,0,0,0,1])
            out_1_1=float(out[0,1,0,0,1])

        div_0_0=out_0_0/x_0_0
        div_1_0=out_1_0/x_1_0
        div_0_1=out_0_1/x_0_1
        div_1_1=out_1_1/x_1_1

        with self.cached_session() as sess:
            self.assertAlmostEqual(div_0_0, div_1_0,places=5)
            self.assertAlmostEqual(div_0_1, div_1_1,places=5)
            
    def test_sSE_3d_excitation_op(self):
        input_shape = (2, 16, 16, 16, 32)
        x = tf.random_uniform(input_shape,seed=0)
        se_layer = SpatialSELayer()
        out_se = se_layer(x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x=sess.run(x)
            x_0_0=float(x[0,0,0,0,0])
            x_0_1=float(x[0,0,0,0,1])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_se)
            out_0_0=float(out[0,0,0,0,0])
            out_0_1=float(out[0,0,0,0,1])

        div_0_0=out_0_0/x_0_0
        div_0_1=out_0_1/x_0_1

        with self.cached_session() as sess:
            self.assertAlmostEqual(div_0_0, div_0_1,places=5)

    def test_cSE_2d_excitation_op(self):
        input_shape = (2, 16, 16, 32)
        x = tf.random_uniform(input_shape,seed=0)
        se_layer = ChannelSELayer()
        out_se = se_layer(x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x=sess.run(x)
            x_0_0=float(x[0,0,0,0])
            x_1_0=float(x[0,1,0,0])
            x_0_1=float(x[0,0,0,1])
            x_1_1=float(x[0,1,0,1])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_se)
            out_0_0=float(out[0,0,0,0])
            out_1_0=float(out[0,1,0,0])
            out_0_1=float(out[0,0,0,1])
            out_1_1=float(out[0,1,0,1])

        div_0_0=out_0_0/x_0_0
        div_1_0=out_1_0/x_1_0
        div_0_1=out_0_1/x_0_1
        div_1_1=out_1_1/x_1_1

        with self.cached_session() as sess:
            self.assertAlmostEqual(div_0_0, div_1_0,places=5)
            self.assertAlmostEqual(div_0_1, div_1_1,places=5)
            
    def test_sSE_2d_excitation_op(self):
        input_shape = (2, 16, 16, 32)
        x = tf.random_uniform(input_shape,seed=0)
        se_layer = SpatialSELayer()
        out_se = se_layer(x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x=sess.run(x)
            x_0_0=float(x[0,0,0,0])
            x_0_1=float(x[0,0,0,1])

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            out = sess.run(out_se)
            out_0_0=float(out[0,0,0,0])
            out_0_1=float(out[0,0,0,1])

        div_0_0=out_0_0/x_0_0
        div_0_1=out_0_1/x_0_1

        with self.cached_session() as sess:
            self.assertAlmostEqual(div_0_0, div_0_1,places=5)

    def test_cSE_pooling_op_error(self):
            with self.cached_session() as sess:
                sess.run(tf.global_variables_initializer())

                with self.assertRaises(ValueError):
                    ChannelSELayer(func='ABC')

    def test_cSE_reduction_ratio_error(self):
        input_shape = (2, 16, 16, 16, 33)
        x = tf.ones(input_shape)
        se_layer = ChannelSELayer()

        with self.assertRaises(ValueError):
            se_layer(x)

if __name__ == "__main__":
    tf.test.main()
