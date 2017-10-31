from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import ImageWindow
from niftynet.layer.base_layer import Layer
from niftynet.layer.grid_warper import AffineGridWarperLayer
from niftynet.layer.resampler import ResamplerLayer


class PairwiseSampler(Layer):
    def __init__(self,
                 reader_0,
                 reader_1,
                 data_param,
                 batch_size,
                 window_per_image):
        Layer.__init__(self, name='pairwise_sampler')
        # reader for the fixed images
        self.reader_0 = reader_0
        # reader for the moving images
        self.reader_1 = reader_1

        # TODO:
        # 0) check the readers should have the same lenght file list
        # 1) detect window shape mismatches or defaulting
        #    windows to the fixed image reader properties
        # 2) reshape images to (supporting multi-modal data)
        #    [batch, x, y, channel] or [batch, x, y, z, channels]
        # 3) infer spatial rank
        self.spatial_rank = 3
        self.window = ImageWindow.from_data_reader_properties(
            self.reader_0.input_sources,
            self.reader_0.shapes,
            self.reader_0.tf_dtypes,
            data_param)
        if self.window.has_dynamic_shapes:
            tf.logging.fatal(
                'Dynamic shapes not supported.\nPlease specify '
                'all spatial dims of the input data, for the '
                'spatial_window_size parameter.')
            raise NotImplementedError
        # TODO: check spatial dims the same across input modalities
        self.image_shape = \
            self.reader_0.shapes['fixed_image'][:self.spatial_rank]
        self.window_size = self.window.shapes['fixed_image']
        pass

    def get_image(self, image_source_type, image_id):
        # returns a random image from either the list of fixed images
        # or the list of moving images
        if image_source_type.startswith('fixed'):
            _, data, _ = self.reader_0(idx=image_id, shuffle=True)
        else:  # image_source_type.startswith('moving'):
            _, data, _ = self.reader_1(idx=image_id, shuffle=True)
        image = data[image_source_type].astype(np.float32)
        image_shape = list(image.shape)
        image = np.reshape(image, image_shape[:self.spatial_rank] + [-1])
        #image = np.swapaxes(image, 0, 1)
        self.image_shape = image.shape  # not thread-safe
        print(image_id)
        return image, np.asarray(image.shape).astype(np.int32)

    def layer_op(self):
        rand_int = tf.random_uniform(
            [], maxval=len(self.reader_0.output_list), dtype=tf.int32)
        # rand_int = np.random.randint(len(self.reader_0.output_list))
        image_0, im_s = tf.py_func(
            self.get_image, ['fixed_image', rand_int], [tf.float32, tf.int32])
        image_1, _ = tf.py_func(
            self.get_image, ['moving_image', rand_int], [tf.float32, tf.int32])
        label_0, _ = tf.py_func(
            self.get_image, ['fixed_label', rand_int], [tf.float32, tf.int32])
        label_1, _ = tf.py_func(
            self.get_image, ['moving_label', rand_int], [tf.float32, tf.int32])

        # TODO preprocessing layer modifying
        #      image shapes will not be supported
        # assuming the same shape across modalities, using the first
        im_s.set_shape((4,))
        image_shape = tf.unstack(im_s)
        #image_shape = [dim for dim in tf.unstack(im_s)]
        #image_shape = [tf.constant(dim) for dim in self.image_shape]
        # image_shape = self.reader_0.output_list[rand_int]['fixed_image'].shape
        # image_0.set_shape(self.image_shape)
        # image_1.set_shape(self.image_shape)
        # label_0.set_shape(self.image_shape)
        # label_1.set_shape(self.image_shape)
        image_to_sample = tf.stack([image_0, image_1, label_0, label_1])

        # TODO affine data augmentation here
        if self.spatial_rank == 3:
            # TODO if no affine augmentation:
            img_spatial_shape = image_shape[:self.spatial_rank]
            win_spatial_shape = [tf.constant(dim) for dim in
                                 self.window_size[:self.spatial_rank]]

            # TODO shifts dtype should be int?
            rand_shift = [
                tf.random_uniform((1,), maxval=tf.to_float(img - win - 1))
                for win, img in zip(win_spatial_shape, img_spatial_shape)]
            # shifting params. in batch size of 4 elements for image_to_sample
            rand_shift = tf.stack(rand_shift, axis=-1)
            rand_shift = tf.tile(rand_shift, (4, 1))
            affine_constraints = ((1.0, 0.0, 0.0, None),
                                  (0.0, 1.0, 0.0, None),
                                  (0.0, 0.0, 1.0, None))
            computed_grid = AffineGridWarperLayer(
                source_shape=(None, None, None),
                output_shape=self.window_size[:self.spatial_rank],
                constraints=affine_constraints)(rand_shift)
            resampler = ResamplerLayer(
                interpolation='linear', boundary='replicate')
            # squeeze image_to_sample to [4, x, y, z, features]
            #image_to_sample = tf.reshape(
            #    image_to_sample, [4] + list(img_spatial_shape) + [-1])
            windows = resampler(image_to_sample, computed_grid)
        return windows

    # overriding input buffers
    def run_threads(self, *args, **argvs):
        # do nothing
        pass

    def close_all(self):
        # do nothing
        pass
