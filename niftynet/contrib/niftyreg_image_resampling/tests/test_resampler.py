from glob import glob
import math
import os.path as osp
import nibabel as nib
import numpy as np
import copy

import tensorflow as tf
import tensorflow.test as tft

from niftynet.contrib.niftyreg_image_resampling.niftyreg_module_loader import get_niftyreg_module


res = get_niftyreg_module()


class ResamplerTest(tft.TestCase):
    """
    Unit test for GPUImageResampling defined in python_wrapper.cpp
    """

    def _get_images(self, do_small=False):
        data_dir = osp.join(osp.dirname(__file__), 'data')
        filename_temp = 'test_image_' + ('vsmall' if do_small else 'large') + '_*.nii'

        for test_img_path in glob(osp.join(data_dir, filename_temp)):
            yield nib.load(test_img_path)

    @staticmethod
    def _dump_image(dst_path, img):
        nib.save(nib.Nifti1Image(img, np.eye(4)),
                 str(dst_path))

    def _test_resampled(self, resampled, floating, d, inter, axis):
        use_nearest = inter == 0
        max_idx_disp = int(round(d)) if use_nearest else int(math.ceil(d))

        self.assertAllEqual(resampled.shape, floating.shape)

        ref = float('nan')*np.zeros_like(resampled)
        size = ref.shape[axis]

        ref_slice = slice(0, size - max_idx_disp)
        flo_slice = slice(max_idx_disp, size)

        base_block = [slice(0, d) for d in floating.shape]
        flo_block = copy.copy(base_block)
        ref_block = copy.copy(base_block)
        flo_block[axis] = flo_slice
        ref_block[axis] = ref_slice

        ref[tuple(ref_block)] = floating[tuple(flo_block)]

        if not use_nearest:
            flo_slice = slice(max_idx_disp - 1, size - 1)
            flo_block = copy.copy(base_block)
            flo_block[axis] = flo_slice
            ref[tuple(ref_block)] += floating[tuple(flo_block)]
            ref /= 2

        mask = np.isfinite(ref + resampled)
        max_err = abs(ref[mask] - resampled[mask]).max()
        self.assertLessEqual(max_err, 1e-2)

        if len(resampled.shape) == 2:
            bdy_size = 1 + max(max_idx_disp, 1)*(inter + 1)\
                *(resampled.shape[0] + resampled.shape[1])
        else:
            bdy_size = 7 + (max(max_idx_disp, 1) + 2)*(inter + 1)\
                *max(resampled.shape)**2

        nof_nans = resampled.size - mask.sum()
        self.assertLess(nof_nans, bdy_size)

    def _make_constant_displacement_image(self, u, axis, image_data):
        nof_dims = len(image_data.shape)
        displacement_shape = list(image_data.shape) \
            + [1]*(3 - nof_dims) + [nof_dims]

        displacement_data = np.zeros(displacement_shape)
        displacement_data[...,axis] = u
        for dd in range(nof_dims):
            idcs = np.arange(displacement_data.shape[dd])
            idcs = idcs.reshape(
                [1]*dd + [displacement_shape[dd]] \
                + [1]*(nof_dims - dd - 1))

            tile_dim = list(displacement_shape[:dd]) + [1] \
                + list(displacement_shape[dd+1:nof_dims])

            idcs = np.tile(idcs, tile_dim)
            if len(idcs.shape) < 3:
                idcs = idcs.reshape(list(idcs.shape) + [1])

            displacement_data[...,dd] += idcs

        return displacement_data

    def _test_resampling(self, use_gpu):
        for floating in self._get_images():
            floating_data = floating.get_data()
            nof_dims = len(floating_data.shape)

            transposed_shape = list(floating_data.shape)
            transposed_shape.reverse()
            image_batch_shape = [1]*2 + transposed_shape
            disp_batch_shape = [1] + [nof_dims] + transposed_shape

            for inter in (0, 1, 3):
                with self.session(use_gpu=use_gpu) as sess:
                    img = tf.placeholder(tf.float32,
                                         shape=image_batch_shape)
                    disp = tf.placeholder(tf.float32,
                                          shape=disp_batch_shape)

                    warped = res.niftyreg_image_resampling(img, disp,
                                                           interpolation=inter)

                    for u in (0.5001, 3.50001):
                        for d in range(nof_dims):
                            displacement_data \
                                = self._make_constant_displacement_image(
                                    u, d, floating_data)

                            # NiftyReg expects displacement components to be
                            # indexed w/ slowest index
                            def _transpose(data):
                                return np.transpose(
                                    data, range(len(data.shape) - 1, -1, -1))

                            resampled_data = sess.run(
                                warped,
                                feed_dict={
                                    img: _transpose(floating_data)\
                                    .reshape(image_batch_shape),
                                    disp: _transpose(displacement_data)\
                                    .reshape(disp_batch_shape),
                                })

                            resampled_data = _transpose(resampled_data).reshape(floating_data.shape)

                            self._test_resampled(resampled_data, floating_data,
                                                 u, inter, d)


    def test_cpu_resampling(self):
        self._test_resampling(False)

    def test_gpu_resampling(self):
        if tft.is_gpu_available(cuda_only=True) and tft.is_built_with_cuda():
            self._test_resampling(True)
        else:
            self.skipTest('No CUDA support available')


if __name__ == '__main__':
    tft.main()
