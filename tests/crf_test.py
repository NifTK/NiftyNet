# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.crf import CRFAsRNNLayer
from niftynet.layer.crf import permutohedral_prepare, permutohedral_compute
from tests.niftynet_testcase import NiftyNetTestCase


class CRFTest(NiftyNetTestCase):
    def test_2d3d_shape(self):
        tf.reset_default_graph()
        I = tf.random_normal(shape=[2, 4, 5, 6, 3])
        U = tf.random_normal(shape=[2, 4, 5, 6, 2])
        crf_layer = CRFAsRNNLayer(T=3)
        crf_layer2 = CRFAsRNNLayer(T=2)
        out1 = crf_layer(I, U)
        out2 = crf_layer2(I[:, :, :, 0, :], out1[:, :, :, 0, :])

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())

            out1, out2 = sess.run([out1, out2])
            U_shape = tuple(U.shape.as_list())
            self.assertAllClose(U_shape, out1.shape)
            U_shape = tuple(U[:, :, :, 0, :].shape.as_list())
            self.assertAllClose(U_shape, out2.shape)

    def test_training_3d(self):
        n_features = 2
        n_classes = 3
        # 4-features
        features = tf.random_normal(shape=[2, 8, 8, 8, n_features])
        # 3-class classification
        logits = tf.random_normal(shape=[2, 8, 8, 8, n_classes])
        # ground truth
        gt = tf.random_uniform(
            shape=[2, 8, 8, 8, n_classes], minval=0, maxval=1)

        crf_layer = CRFAsRNNLayer()
        smoothed_logits = crf_layer(features, logits)
        loss = tf.reduce_mean(tf.abs(smoothed_logits - gt))
        opt = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            params = sess.run(tf.trainable_variables())
            for param in params:
                if param.shape == (n_classes, n_classes):
                    self.assertAllClose(param, -1.0 * np.eye(n_classes))
            sess.run(opt)
            params_1 = sess.run(tf.trainable_variables())
            self.assertGreater(np.sum(np.abs(params_1[0] - params[0])), 0.0)

    def test_training_2d(self):
        batch_size = 1
        n_features = 2
        n_classes = 3
        # 2-features
        features = tf.random_normal(shape=[batch_size, 8, 8, n_features])
        # 3-class classification
        logits = tf.random_normal(shape=[batch_size, 8, 8, n_classes])
        # ground truth
        gt = tf.random_uniform(
            shape=[batch_size, 8, 8, n_classes], minval=0, maxval=1)

        crf_layer = CRFAsRNNLayer(
            w_init=[[1] * n_classes, [1] * n_classes],
            mu_init=np.eye(n_classes),
            T=2)
        smoothed_logits = crf_layer(features, logits)
        pred = tf.nn.softmax(smoothed_logits)
        loss = tf.reduce_mean(tf.abs(smoothed_logits - gt))
        opt = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            params = sess.run(tf.trainable_variables())
            for param in params:
                if param.shape == (n_classes, n_classes):
                    self.assertAllClose(param, np.eye(n_classes))
            sess.run(opt)
            params_1 = sess.run(tf.trainable_variables())
            print(params_1)
            self.assertGreater(np.sum(np.abs(params_1[0] - params[0])), 0.0)

    def test_training_4d(self):
        sp = 8
        batch_size = 2
        n_features = 2
        n_classes = 3
        # 2-features
        features = tf.random_normal(
            shape=[batch_size, sp, sp, sp, sp, n_features])
        # 3-class classification
        logits = tf.random_normal(
            shape=[batch_size, sp, sp, sp, sp, n_classes])
        # ground truth
        gt = tf.random_uniform(
            shape=[batch_size, sp, sp, sp, sp, n_classes], minval=0, maxval=1)

        with tf.device('/cpu:0'):
            crf_layer = CRFAsRNNLayer(
                w_init=[[1] * n_classes, [1] * n_classes],
                mu_init=np.eye(n_classes),
                T=2)
            smoothed_logits = crf_layer(features, logits)
        loss = tf.reduce_mean(tf.abs(smoothed_logits - gt))
        opt = tf.train.GradientDescentOptimizer(0.5).minimize(
            loss, colocate_gradients_with_ops=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            params = sess.run(tf.trainable_variables())
            for param in params:
                if param.shape == (n_classes, n_classes):
                    self.assertAllClose(param, np.eye(n_classes))
            sess.run(opt)
            params_1 = sess.run(tf.trainable_variables())
            self.assertGreater(np.sum(np.abs(params_1[0] - params[0])), 0.0)

    def test_batch_mix(self):
        feat = tf.random.uniform(shape=[2, 64, 5])
        desc = tf.ones(shape=[1, 64, 1])
        desc_ = tf.zeros(shape=[1, 64, 1])
        desc = tf.concat([desc, desc_], axis=0)
        barycentric, blur_neighbours1, blur_neighbours2, indices = permutohedral_prepare(feat)
        sliced = permutohedral_compute(desc,
                          barycentric,
                          blur_neighbours1,
                          blur_neighbours2,
                          indices,
                          "test",
                          True)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sliced_np = sess.run(sliced)
            self.assertAllClose(sliced_np[1:], np.zeros(shape=[1, 64, 1]))



if __name__ == "__main__":
    tf.test.main()
