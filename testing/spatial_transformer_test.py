from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.contrib.layers.python.layers import regularizers

from layer.convolution import ConvLayer
from layer.spatial_transformer import ResamplerLayer, AffineGridWarperLayer


class ResamplerTest(tf.test.TestCase):
    def get_3d_input1(self):
        return tf.expand_dims(tf.constant([[[[1,2],[3,4]],[[5,6],[7,8]]],[[[9,10],[11,12]],[[13,14],[15,16]]]],dtype=tf.float32),4)
    def get_identity_warp(self):
        params=tf.tile(tf.constant([[1,0,0,0,0,1,0,0,0,0,1,0]],dtype=tf.float32),[sz[0],1])
        grid=AffineGridWarperLayer(source_shape=sz[1:4], output_shape=sz[1:4])
        return grid,params
    def test_resampler_2d_linear_correctness(self):
        input=self.get_3d_input1()
        grid=tf.constant([[[.25,.25,.25],[.25,.75,.25]],[[.75,.25,.25],[.25,.25,.75]]],dtype=tf.float32)
        resampler=ResamplerLayer()
        out=resampler(input,grid)
        with self.test_session() as sess:
          sess.run(tf.global_variables_initializer())
          out_value = sess.run(out)
          self.assertAllClose([[[2.75],[3.75]],[[12.75],[11.25]]],out_value)
    def test_resampler_2d_nearest_correctness(self):
        input=self.get_3d_input1()
        grid=tf.constant([[[.25,.25,.25],[.25,.75,.25]],[[.75,.25,.25],[.25,.25,.75]]],dtype=tf.float32)
        resampler=ResamplerLayer(interpolation='NEAREST')
        out=resampler(input,grid)
        with self.test_session() as sess:
          sess.run(tf.global_variables_initializer())
          out_value = sess.run(out)
          self.assertAllClose([[[1],[3]],[[13],[10]]],out_value)


if __name__ == "__main__":
    tf.test.main()
