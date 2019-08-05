from niftynet.contrib.niftyreg_image_resampling.niftyreg_image_resampling import NiftyregImageResamplingLayer
from niftynet.contrib.niftyreg_image_resampling.tests.test_resampler import ResamplerTest

import numpy as np
import tensorflow as tf
import tensorflow.test as tft

class WrapperResamplerTest(ResamplerTest):
    """
    Unit test for NiftyregImageResamplingLayer
    """

    INTERPOLATIONS = ((0, 'NEAREST'),
                      (1, 'LINEAR'),
                      (3, 'BSPLINE'))

    def _test_differential(self, use_gpu, interpolations):
        for floating in self._get_images(True):
            floating_data = floating.get_data()
            nof_dims = len(floating_data.shape)
            nof_mods = 2

            if nof_dims == 3:
                # 3D test takes forever (at least with debug builds)
                continue

            floating_shape = list(floating_data.shape)
            image_batch_shape = [1] + floating_shape + [nof_mods]
            disp_batch_shape = [1] + floating_shape + [nof_dims]

            disp_template \
                = self._make_constant_displacement_image(0, 0,
                                                         floating_data)
            u = 2.5001

            floating_data = np.stack([floating_data]*nof_mods, axis=-1)
            for m in range(nof_mods):
                floating_data[...,m] = (1 + m)*floating_data[...,m]

            for iname in interpolations:
                d = 1
                with self.session(use_gpu=use_gpu) as sess:
                    img = tf.constant(
                        floating_data.reshape(image_batch_shape),
                        dtype=tf.float32)
                    disp = tf.constant(u,
                                       dtype=tf.float32)

                    base_field = tf.constant(disp_template\
                                             .reshape(disp_batch_shape),
                                             dtype=tf.float32)

                    disp_field = []
                    for i in range(nof_dims):
                        if i == d:
                            disp_field.append(base_field[...,i] + disp)
                        else:
                            disp_field.append(base_field[...,i])
                    disp_field = tf.stack(disp_field, axis=-1)

                    warped = NiftyregImageResamplingLayer(interpolation=iname,
                                                          boundary='ZERO')
                    warped = warped(img, disp_field)

                    tgrad, refgrad = tft.compute_gradient(
                        disp,
                        (),
                        warped,
                        tuple(image_batch_shape))

                    error = np.power(tgrad - refgrad, 2).sum()
                    refmag = np.power(refgrad, 2).sum()

                    self.assertLessEqual(error, 1e-2*refmag)

    def test_cpu_differential(self):
        self._test_differential(False, ('LINEAR', 'BSPLINE'))

    def test_gpu_differential(self):
        if tft.is_gpu_available(cuda_only=True) and tft.is_built_with_cuda():
            self._test_differential(True, ('LINEAR', 'BSPLINE'))
        else:
            self.skipTest('No CUDA support available')

    def test_image_gradient(self):
        for img in self._get_images(True):
            basedata = img.get_data()

            imgshape = list(basedata.shape)
            nof_dims = len(imgshape)
            for nof_mods in range(1, 3):
                image_batch_shape = [1] + imgshape + [nof_mods]
                disp_batch_shape = [1] + imgshape + [nof_dims]

                multimod_data = []
                for j in range(nof_mods):
                    multimod_data.append((j + 1)*basedata)
                imgdata = np.stack(multimod_data, axis=-1)

                for _, inter in self.INTERPOLATIONS:
                    for bdy in ('REPLICATE', 'ZERO', 'SYMMETRIC'):
                        u = 3.5001
                        d = 1
                        disp = self._make_constant_displacement_image(
                            u, d, imgdata[...,0].reshape(imgshape))

                        with self.session() as sess:
                            tfimg = tf.constant(
                                imgdata.reshape(image_batch_shape),
                                dtype=tf.float32)
                            tfdisp = tf.constant(
                                disp.reshape(disp_batch_shape),
                                dtype=tf.float32)

                            warped = NiftyregImageResamplingLayer(interpolation=inter,
                                                                  boundary=bdy)
                            warped = warped(tfimg, tfdisp)
                            dummy_cost = tf.reduce_mean(tf.pow(warped, 2))

                            tgrad, refgrad = tft.compute_gradient(
                                tfimg,
                                image_batch_shape,
                                dummy_cost,
                                ())

                            error = np.power(tgrad - refgrad, 2).sum()
                            refmag = np.power(refgrad, 2).sum()

                            self.assertLessEqual(error, 5e-2*refmag)

    def _test_resampling(self, use_gpu):
        for floating in self._get_images():
            floating_data = floating.get_data()
            nof_dims = len(floating_data.shape)

            floating_shape = list(floating_data.shape)
            if floating_shape[-1] == 1:
                floating_shape = floating_shape[:-1]
            image_batch_shape = [1] + floating_shape + [1]
            disp_batch_shape = [1] + floating_shape + [nof_dims]

            for code, inter in self.INTERPOLATIONS:
                with self.session(use_gpu=use_gpu) as sess:
                    img = tf.placeholder(tf.float32,
                                         shape=image_batch_shape)
                    disp = tf.placeholder(tf.float32,
                                          shape=disp_batch_shape)

                    warped = NiftyregImageResamplingLayer(interpolation=inter,
                                                          boundary='NAN')
                    warped = warped(img, disp)

                    for u in (0.5001, 3.50001):
                        for d in range(nof_dims):
                            displacement_data \
                                = self._make_constant_displacement_image(
                                    u, d, floating_data)

                            resampled_data = sess.run(
                                warped,
                                feed_dict={
                                    img: floating_data\
                                    .reshape(image_batch_shape),
                                    disp: displacement_data\
                                    .reshape(disp_batch_shape),
                                })

                            resampled_data = resampled_data.reshape(floating_shape)

                            self._test_resampled(resampled_data, floating_data,
                                                 u, code, d)


    def test_cpu_resampling(self):
        self._test_resampling(False)

    def test_gpu_resampling(self):
        if tft.is_gpu_available(cuda_only=True) and tft.is_built_with_cuda():
            self._test_resampling(True)
        else:
            self.skipTest('No CUDA support available')


if __name__ == '__main__':
    tft.main()
