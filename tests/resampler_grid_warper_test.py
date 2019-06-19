from __future__ import absolute_import, print_function, division

import base64

import numpy as np
import tensorflow as tf

from niftynet.layer.grid_warper import AffineGridWarperLayer
from niftynet.layer.resampler import ResamplerLayer
from tests.niftynet_testcase import NiftyNetTestCase

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

test_case_2d_target = {
    # [[0.96592583, -0.25881905,  2.34314575],
    # [0.25881905,  0.96592583, -1.79795897]]
    'data': "////19jdbXKIZFl3TC5GVzM1yaKR/9vN/ODU7vnR2v/M0v7N9f/2///9////"
            "////////pau3Vlx2aF90aFFXkW5a8c+s/uTD6+a8sOiPauRTR/M9a/102P7n"
            "/v7///v////9dYGPXmB1cWVzX0c7v5dz/t+z++q8wN+RWdQ9E98ECO8DINkj"
            "keSW//76/+z49vf5YmR7X1duc11pdFRF6cGe/+fD9OvEoNyENuIuAv0AAP8B"
            "Gu4Qd9thx8mi07Gly8nWZFmBc1l8bUtck3Jp//Te//Ll/f7wxP7DKdIvAPUA"
            "BP8CE9wAVKspbWguWjgToZq8bVaOeE5+b0NcqoSD/vTo//T4/fP74f7of+19"
            "KugkLPMeTNUvjrhWclgnlmhHc2yYb1SLdkt5jGF1u6OZ5uDU9/L4+/T88fni"
            "zPirletwmfF21P6ox7mKhlI8klVDYmGAblp/eVJve1db1ci36+/e8PTz9Ozq"
            "+OjO+fC18Pas3eKg+vDM06WVj1BHllZNXWF6aVxwbFFYkXps/fPY+v7v+P35"
            "+fLq9+LF/+m3/eOw3L6a/+DO2qmbg0k7lVxLX2B/aF1tZVBLuaOM//Db/fr1"
            "+Pn7//309+zM9+K19dyz5suu/N3IwpmDYjcXkGdJYFmFa19zWEE52ryk/+zd"
            "/OPm/O/2/fPp/PLP8uS39uK9/+7Q6tC1lHNUXTkVr5aAUUVtemR5XT8368Ww"
            "/9zM987I//Ho/+zM8OKx+ey3896z/+fDwJ9+f1o/gFtA2tDGbVlyVTQ/dlBH"
            "6sau/uTL/9fB/uK9/+yx+eai/+qr/ee247OLilk7gk88kWFX+Pf13MPJj2Zk"
            "kmZZ68as/eLE+eG9+uWw+uWk/OSk/uWs4rqJn2w/iE8xkVVKpXNy/////vj4"
            "7NDPuJKF79G58ebK8e3I9fPD+++9/vHO/+vQr45vcEgkiVg4lV1QxaOi////"
            "//////////////78+fnv9Pni8PfY/frn/Pj5/f3/9+7lp5Z8eFo4gVdB5drW"
            "////".encode("ASCII"),
    'shape': (16, 16, 3)
}


def get_2d_images(test_case):
    try:
        out = base64.decodebytes(test_case['data'])
    except AttributeError:
        out = base64.decodestring(test_case['data'])
    out = np.frombuffer(out, dtype=np.uint8)
    out = out.reshape(test_case['shape'])
    return out, out.shape


def get_multiple_2d_images():
    image_1, shape = get_2d_images(test_case_2d_1)
    image_2 = image_1[::-1, ::-1]
    image_3 = image_1[::-1, ]
    image_4 = image_1[:, ::-1, ]
    return np.stack([image_1, image_2, image_3, image_4]), [4] + list(shape)


def get_multiple_2d_rotated_targets():
    image_1, shape = get_2d_images(test_case_2d_target)
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


def get_multiple_3d_images():
    image_1, shape = get_2d_images(test_case_2d_1)
    image_2 = image_1[::-1, ::-1]
    image_3 = image_1[::-1, ]
    image_4 = image_1[:, ::-1, ]

    image_2d = np.stack([image_1, image_2, image_3, image_4])
    image_3d = np.expand_dims(image_2d, axis=1)
    image_3d = np.concatenate([image_3d, image_3d], axis=1)
    return image_3d, image_3d.shape


def get_multiple_3d_targets():
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
    test_target = np.expand_dims(test_target, axis=1)
    test_target = np.concatenate([test_target, test_target], axis=1)
    return test_target, test_target.shape


def get_3d_input1():
    test_case = tf.constant(
        [[[[1, 2, -1], [3, 4, -2]], [[5, 6, -3], [7, 8, -4]]],
         [[[9, 10, -5], [11, 12, -6]], [[13, 14, -7], [15, 16, -8]]]],
        dtype=tf.float32)
    return tf.expand_dims(test_case, 4)


class ResamplerGridWarperTest(NiftyNetTestCase):
    def _test_correctness(
            self, inputs, grid, interpolation, boundary, expected_value):
        resampler = ResamplerLayer(
            interpolation=interpolation, boundary=boundary)
        out = resampler(inputs, grid)
        with self.cached_session() as sess:
            out_value = sess.run(out)
            self.assertAllClose(expected_value, out_value)

    def test_combined(self):
        expected = [[[[[1], [-1]], [[3], [-2]]],
                     [[[5], [-3]], [[7], [-4]]]],
                    [[[[9.5], [-5]], [[11.5], [-6]]],
                     [[[13.5], [-7]], [[15.5], [-8]]]]]
        affine_grid = AffineGridWarperLayer(source_shape=(2, 2, 3),
                                            output_shape=(2, 2, 2))
        test_grid = affine_grid(
            tf.constant([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                         [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, .5]],
                        dtype=tf.float32))
        self._test_correctness(inputs=get_3d_input1(),
                               grid=test_grid,
                               interpolation='idw',
                               boundary='replicate',
                               expected_value=expected)


class image_test(NiftyNetTestCase):
    def _test_grads_images(self,
                           interpolation='linear',
                           boundary='replicate',
                           ndim=2):
        if ndim == 2:
            test_image, input_shape = get_multiple_2d_images()
            test_target, target_shape = get_multiple_2d_targets()
            identity_affine = [[1., 0., 0., 0., 1., 0.]] * 4
        else:
            test_image, input_shape = get_multiple_3d_images()
            test_target, target_shape = get_multiple_3d_targets()
            identity_affine = [[1., 0., 0., 0., 1., 0.,
                                1., 0., 0., 0., 1., 0.]] * 4
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
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            init_val, affine_val = sess.run([diff, affine_var])
            for _ in range(5):
                _, diff_val, affine_val = sess.run([opt, diff, affine_var])
                print('{}, {}'.format(diff_val, affine_val[0]))
            self.assertGreater(init_val, diff_val)

    def test_2d_linear_replicate(self):
        self._test_grads_images('linear', 'replicate')

    def test_2d_idw_replicate(self):
        self._test_grads_images('idw', 'replicate')

    def test_2d_linear_circular(self):
        self._test_grads_images('linear', 'circular')

    def test_2d_idw_circular(self):
        self._test_grads_images('idw', 'circular')

    def test_2d_linear_symmetric(self):
        self._test_grads_images('linear', 'symmetric')

    def test_2d_idw_symmetric(self):
        self._test_grads_images('idw', 'symmetric')

    def test_3d_linear_replicate(self):
        self._test_grads_images('linear', 'replicate', ndim=3)

    def test_3d_idw_replicate(self):
        self._test_grads_images('idw', 'replicate', ndim=3)

    def test_3d_linear_circular(self):
        self._test_grads_images('linear', 'circular', ndim=3)

    def test_3d_idw_circular(self):
        self._test_grads_images('idw', 'circular', ndim=3)

    def test_3d_linear_symmetric(self):
        self._test_grads_images('linear', 'symmetric', ndim=3)

    def test_3d_idw_symmetric(self):
        self._test_grads_images('idw', 'symmetric', ndim=3)


class image_2D_test_converge(NiftyNetTestCase):
    def _test_simple_2d_images(self,
                               interpolation='linear',
                               boundary='replicate'):
        # rotating around the center (8, 8) by 15 degree
        expected = [[0.96592583, -0.25881905, 2.34314575],
                    [0.25881905, 0.96592583, -1.79795897]]
        expected = np.asarray(expected).flatten()
        test_image, input_shape = get_multiple_2d_images()
        test_target, target_shape = get_multiple_2d_rotated_targets()

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
        learning_rate = 0.05
        if(interpolation == 'linear') and (boundary == 'zero'):
            learning_rate = 0.0003
        optimiser = tf.train.AdagradOptimizer(learning_rate)
        grads = optimiser.compute_gradients(diff)
        opt = optimiser.apply_gradients(grads)
        with self.cached_session() as sess:
            sess.run(tf.global_variables_initializer())
            init_val, affine_val = sess.run([diff, affine_var])
            # compute the MAE between the initial estimated parameters and the expected parameters
            init_var_diff = np.sum(np.abs(affine_val[0] - expected))
            for it in range(500):
                _, diff_val, affine_val = sess.run([opt, diff, affine_var])
                # print('{} diff: {}, {}'.format(it, diff_val, affine_val[0]))
            # import matplotlib.pyplot as plt
            # plt.figure()
            # plt.imshow(test_target[0])
            # plt.draw()

            # plt.figure()
            # plt.imshow(sess.run(new_image).astype(np.uint8)[0])
            # plt.draw()

            # plt.show()
            self.assertGreater(init_val, diff_val)
            # compute the MAE between the final estimated parameters and the expected parameters
            var_diff = np.sum(np.abs(affine_val[0] - expected))
            self.assertGreater(init_var_diff, var_diff)
            print('{} {} -- diff {}'.format(
                interpolation, boundary, var_diff))
            print('{}'.format(affine_val[0]))

    def test_2d_linear_zero_converge(self):
        self._test_simple_2d_images('linear', 'zero')

    def test_2d_linear_replicate_converge(self):
        self._test_simple_2d_images('linear', 'replicate')

    def test_2d_idw_replicate_converge(self):
        self._test_simple_2d_images('idw', 'replicate')

    def test_2d_linear_circular_converge(self):
        self._test_simple_2d_images('linear', 'circular')

    def test_2d_idw_circular_converge(self):
        self._test_simple_2d_images('idw', 'circular')

    def test_2d_linear_symmetric_converge(self):
        self._test_simple_2d_images('linear', 'symmetric')

    def test_2d_idw_symmetric_converge(self):
        self._test_simple_2d_images('idw', 'symmetric')


if __name__ == "__main__":
    tf.test.main()
