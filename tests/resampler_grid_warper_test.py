from __future__ import absolute_import, print_function, division

import base64

import numpy as np
import tensorflow as tf

from niftynet.layer.grid_warper import AffineGridWarperLayer
from niftynet.layer.resampler import ResamplerLayer

test_case_2d_1 = {
    'data': "+/b9/+3/377dpX+Mxp+Y/9nT/d/X6vfMuf+hX/hSY/1pvf/P9/z//+///+7z"
            "//ve19noiHuXVjlVSCUpwpyH/9i/9+LDwuufS84yGOYKGOgYQspG2v7Q/uXg"
            "07aonZBtS1NqWVRycl9zZEY86sSf/+u/7uezlNlvIdYPA/8AAP8AK+MfgMRd"
            "f3JGVzYTdV0xW2d9Y2N7c2NuZEgz58CV/+S66OS1jdt2KOclAP8AAP8AFtkB"
            "V6Ema1wjkmZDkmdFXGd5XltwdWFqdldF8c2r/+/V//7szP/JOs9AC+gNGvkS"
            "P9YlrNp4fl41kVdDj1ZDYWN8ZFdzblFjfVpU/+/a//Hp/e718P/2v/+8bOdb"
            "auVOtv6Q9fW/om9eiEg/oGFYXFR9e2GOdEttbkZO7tPI//v2//P/+/f47PjQ"
            "3Pmn3fmi3eGm/+rRyZCHhEg9l19Oal2TbU6HeUp2lm17x7Wn5eXZ7e7w9evp"
            "+OXH/+yz+uWs3b+b/9/N3a6ebj8lg1Y1ZFyNcFWIelB0fFde2Mu48fjm+f/7"
            "+PPt9uLH/+m6/+W24cSk/+TNz62SUS0LeVYuYGGAa1x9dFRpdldS9OXO/P3r"
            "8vb1//78//bg8OG28d6z/OjH/+nLwqWHbksrh2JFWmB6ZWB2aVVedl9R893F"
            "//Hl//r/++/z//Xh/PDG9Oa38Nqx/uC+ontcek04kWFVYWWKX1x5bWBqZE0/"
            "8dO7/+re89HS//Xx/uvK7+Cp/++1/+u74rWMhE8vilJBk1lYWVmNX1iCbF1y"
            "VToz58Gs/9rH/tLF/+DG/+y2/uej/+Ki/92pq3hLjVcxlFtHkVZSbGmYTkNt"
            "gmqCWzg22K2a/+TL/93C++C1++eq+OOi/+q489GsfVk3dlArkGRJkGFR3dnw"
            "lIadXT5NSiEdvpOA/93C8+DA8+rB+PLA/PDI//fn//v47eHVpph9cVo7ZkYt"
            "/f37//f678zQxpeRrYJx993G8OvO7vTQ8PbU/fvs/Pj/9/n/9///+P/t8OnM"
            "4s2u".encode('ascii'),
    'shape': (16, 16, 3)
}


def get_2d_images():
    try:
        out = base64.decodebytes(test_case_2d_1['data'])
    except AttributeError:
        out = base64.decodestring(test_case_2d_1['data'])
    out = np.frombuffer(out, dtype=np.uint8)
    out = out.reshape(test_case_2d_1['shape'])
    return out, out.shape

def get_multiple_2d_images():
    image_1, shape = get_2d_images()
    image_2 = image_1[::-1, ::-1]
    image_3 = image_1[::-1, ]
    image_4 = image_1[:, ::-1, ]
    return np.stack([image_1, image_2, image_3, image_4]), [4] + list(shape)

def get_multiple_2d_targets():
    test_image, input_shape = get_multiple_2d_images()
    test_target = np.array(test_image)

    test_target[0] = test_target[0, ::-1]
    test_target[1] = test_target[1, :, ::-1]
    test_target[2] = test_target[2, ::-1, ::-1]

    factor = 1.5
    shape = input_shape[:]
    shape[1] = np.floor(input_shape[1] * factor).astype(np.int)
    shape[2] = np.floor(input_shape[2] * factor).astype(np.int)

    from scipy.ndimage import zoom
    zoomed_target = []
    for img in test_target:
        zoomed_target.append(zoom(img, [factor, factor, 1]))
    test_target = np.stack(zoomed_target, axis=0).astype(np.uint8)
    return test_target, shape


class ResamplerGridWarperTest(tf.test.TestCase):
    def get_3d_input1(self):
        test_case = tf.constant(
            [[[[1, 2, -1], [3, 4, -2]], [[5, 6, -3], [7, 8, -4]]],
             [[[9, 10, -5], [11, 12, -6]], [[13, 14, -7], [15, 16, -8]]]],
            dtype=tf.float32)
        return tf.expand_dims(test_case, 4)

    def _test_correctness(
            self, input, grid, interpolation, boundary, expected_value):
        resampler = ResamplerLayer(interpolation=interpolation,
                                   boundary=boundary)
        out = resampler(input, grid)
        with self.test_session() as sess:
            out_value = sess.run(out)
            self.assertAllClose(expected_value, out_value)

    def test_combined(self):
        expected = [[[[[1], [2]], [[3], [4]]],
                     [[[5], [6]], [[7], [8]]]],
                    [[[[9.5], [2.5]], [[11.5], [3]]],
                     [[[13.5], [3.5]], [[15.5], [4]]]]]
        affine_grid = AffineGridWarperLayer(source_shape=(2, 2, 3),
                                            output_shape=(2, 2, 2))
        test_grid = affine_grid(
            tf.constant([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                         [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, .5]],
                        dtype=tf.float32))
        self._test_correctness(input=self.get_3d_input1(),
                               grid=test_grid,
                               interpolation='idw',
                               boundary='replicate',
                               expected_value=expected)


class image_2D_test(tf.test.TestCase):
    def _test_grads_2d_images(self,
                       interpolation='linear',
                       boundary='replicate'):
        test_image, input_shape = get_multiple_2d_images()
        test_target, target_shape = get_multiple_2d_targets()

        identity_affine = [[1., 0., 0., 0., 1., 0.],
                           [1., 0., 0., 0., 1., 0.],
                           [1., 0., 0., 0., 1., 0.],
                           [1., 0., 0., 0., 1., 0.]]
        affine_var = tf.get_variable('affine', initializer=identity_affine)
        grid = AffineGridWarperLayer(source_shape=input_shape[1:-1],
                                     output_shape=target_shape[1:-1],
                                     constraints=None)
        warp_coords = grid(affine_var)
        resampler = ResamplerLayer(interpolation, boundary=boundary)
        new_image = resampler(tf.constant(test_image, dtype=tf.float32),
                              warp_coords)

        diff = tf.reduce_mean(tf.squared_difference(
            new_image, tf.constant(test_target, dtype=tf.float32)))
        optimiser = tf.train.AdagradOptimizer(0.01)
        grads = optimiser.compute_gradients(diff)
        opt = optimiser.apply_gradients(grads)
        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            init_val, affine_val = sess.run([diff, affine_var])
            for _ in range(5):
                _, diff_val, affine_val = sess.run([opt, diff, affine_var])
                print('{}, {}'.format(diff_val, affine_val[0]))
            self.assertGreater(init_val, diff_val)

    def test_2d_linear_replicate(self):
        self._test_grads_2d_images('linear', 'replicate')

    def test_2d_idw_replicate(self):
        self._test_grads_2d_images('idw', 'replicate')

    def test_2d_linear_circular(self):
        self._test_grads_2d_images('linear', 'circular')

    def test_2d_idw_circular(self):
        self._test_grads_2d_images('idw', 'circular')

    def test_3d_linear_symmetric(self):
        self._test_grads_2d_images('linear', 'symmetric')

    def test_2d_idw_symmetric(self):
        self._test_grads_2d_images('idw', 'symmetric')

if __name__ == "__main__":
    tf.test.main()
