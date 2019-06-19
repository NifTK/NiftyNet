from __future__ import absolute_import, print_function

import tensorflow as tf
from niftynet.layer.crop import CropLayer
from tests.niftynet_testcase import NiftyNetTestCase

class CropTest(NiftyNetTestCase):
    def test_3d_shape(self):
        input_shape = (2, 16, 16, 16, 8)
        test_border = 3
        x = tf.ones(input_shape)

        crop_layer = CropLayer(border=test_border)
        out_crop = crop_layer(x)
        print(crop_layer)

        input_shape = (2, 7, 7, 7, 8)
        test_border = 3
        x = tf.ones(input_shape)

        crop_layer = CropLayer(border=test_border)
        out_crop_1 = crop_layer(x)
        print(crop_layer)

        with self.cached_session() as sess:
            out = sess.run(out_crop)
            out_1 = sess.run(out_crop_1)
            self.assertAllClose((2, 10, 10, 10, 8), out.shape)
            self.assertAllClose((2, 1, 1, 1, 8), out_1.shape)

    def test_2d_shape(self):
        input_shape = (2, 16, 16, 8)
        test_border = 3
        x = tf.ones(input_shape)

        crop_layer = CropLayer(border=test_border)
        out_crop = crop_layer(x)
        print(crop_layer)

        input_shape = (2, 7, 7, 8)
        test_border = 3
        x = tf.ones(input_shape)

        crop_layer = CropLayer(border=test_border)
        out_crop_1 = crop_layer(x)
        print(crop_layer)

        with self.cached_session() as sess:
            out = sess.run(out_crop)
            out_1 = sess.run(out_crop_1)
            self.assertAllClose((2, 10, 10, 8), out.shape)
            self.assertAllClose((2, 1, 1, 8), out_1.shape)


if __name__ == "__main__":
    tf.test.main()
