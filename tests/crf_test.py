# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.layer.crf import CRFAsRNNLayer

class CRFTest(tf.test.TestCase):
    def test_2d3d_shape(self):
        tf.reset_default_graph()
        I = tf.random_normal(shape=[2,4,5,6,3])
        U = tf.random_normal(shape=[2,4,5,6,2])
        crf_layer = CRFAsRNNLayer()
        crf_layer2 = CRFAsRNNLayer()
        out1 = crf_layer(I, U)
        out2 = crf_layer2(I[:,:,:,0,:], out1[:,:,:,0,:])

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())

            out1,out2 = sess.run([out1,out2])
            U_shape = tuple(U.shape.as_list())
            self.assertAllClose(U_shape, out1.shape)
            U_shape = tuple(U[:,:,:,0,:].shape.as_list())
            self.assertAllClose(U_shape, out2.shape)


if __name__ == "__main__":
    tf.test.main()
