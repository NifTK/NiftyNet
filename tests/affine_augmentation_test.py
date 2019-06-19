# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.affine_augmentation import AffineAugmentationLayer
from tests.niftynet_testcase import NiftyNetTestCase

class RandRotationTest(NiftyNetTestCase):
    def get_3d_image(self):
        image_shape = (1, 200, 200, 1)
        elements = tf.range(np.prod(image_shape), dtype=tf.float32)
        image = tf.reshape(elements, image_shape)
        return image

    def get_2d_image(self):
        image_shape = (2, 50, 50, 3)
        elements = tf.range(np.prod(image_shape), dtype=tf.float32)
        image = tf.reshape(elements, image_shape)
        return image

    def test_2d_shape(self):
        augment_layer = AffineAugmentationLayer(0.2)
        input_tensor = self.get_2d_image()
        deformed = augment_layer(input_tensor)

        inverse_augment_layer = augment_layer.inverse()
        inverse = inverse_augment_layer(deformed)

        with self.cached_session() as sess:
            test_out = sess.run([input_tensor, deformed, inverse])
            original, deformed_image, resumed_image = test_out
            to_compare = resumed_image > 0
            original = original[to_compare]
            resumed_image = resumed_image[to_compare]

            correct = np.sum(np.abs(original - resumed_image) < 1.0)
            ratio = float(correct) / float(original.size)
            print(ratio)
            self.assertGreaterEqual(ratio, 0.9)

    def test_3d_shape(self):
        augment_layer = AffineAugmentationLayer(0.2)
        input_tensor = self.get_3d_image()
        deformed = augment_layer(input_tensor)

        inverse_augment_layer = augment_layer.inverse()
        inverse = inverse_augment_layer(deformed)

        # with tf.Session() as sess:
        with self.cached_session() as sess:
            test_out = sess.run([input_tensor, deformed, inverse])
            original, deformed_image, resumed_image = test_out
            to_compare = resumed_image > 0
            original = original[to_compare]
            resumed_image = resumed_image[to_compare]

            correct = np.sum(np.abs(original - resumed_image) < 1.0)
            ratio = float(correct) / float(original.size)
            print(ratio)
            self.assertGreaterEqual(ratio, 0.95)


if __name__ == "__main__":
    tf.test.main()
