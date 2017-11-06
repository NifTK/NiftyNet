from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.image_window import ImageWindow
from niftynet.layer.base_layer import Layer
from niftynet.layer.grid_warper import AffineGridWarperLayer
from niftynet.layer.resampler import ResamplerLayer

from tensorflow.contrib.data.python.ops.dataset_ops import Dataset


class PairwiseSampler(Layer):
    def __init__(self,
                 reader_0,
                 reader_1,
                 data_param,
                 batch_size=1,
                 window_per_image=2):
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

        self.batch_size = batch_size
        self.window_per_image = window_per_image
        if self.window.has_dynamic_shapes:
            tf.logging.fatal('Dynamic shapes not supported.\nPlease specify '
                             'all spatial dims of the input data, for the '
                             'spatial_window_size parameter.')
            raise NotImplementedError
        # TODO: check spatial dims the same across input modalities
        self.image_shape = \
            self.reader_0.shapes['fixed_image'][:self.spatial_rank]
        self.window_size = self.window.shapes['fixed_image']


        n_subjects = len(self.reader_0.output_list)
        rand_ints = np.random.randint(n_subjects, size=[n_subjects])
        image_dataset = Dataset.from_tensor_slices(rand_ints)
        image_dataset = image_dataset.map(
            lambda image_id: tuple(tf.py_func(self.get_pairwise_inputs,
                                              [image_id],
                                              [tf.float32, tf.int32])))
            #num_parallel_calls=4)
        image_dataset = image_dataset.repeat() # num_epochs can be param
        image_dataset = image_dataset.shuffle(buffer_size=batch_size*20)
        image_dataset = image_dataset.batch(batch_size)
        self.iterator = image_dataset.make_initializable_iterator()

    def get_pairwise_inputs(self, image_id):
        fixed_image, _ = self.get_image('fixed_image', image_id)
        fixed_label, _ = self.get_image('fixed_label', image_id)
        moving_image, _ = self.get_image('moving_image', image_id)
        moving_label, _ = self.get_image('moving_label', image_id)
        images = [fixed_image, fixed_label, moving_image, moving_label]
        images = np.concatenate(images, axis=-1)
        images_shape = np.asarray(images.shape).T.astype(np.int32)
        return images, images_shape

    def get_image(self, image_source_type, image_id):
        # returns a random image from either the list of fixed images
        # or the list of moving images
        try:
            image_source_type = image_source_type.decode()
        except:
            pass
        if image_source_type.startswith('fixed'):
            _, data, _ = self.reader_0(idx=image_id, shuffle=True)
        else:  # image_source_type.startswith('moving'):
            _, data, _ = self.reader_1(idx=image_id, shuffle=True)
        image = np.asarray(data[image_source_type]).astype(np.float32)
        image_shape = list(image.shape)
        image = np.reshape(image, image_shape[:self.spatial_rank] + [-1])
        image_shape = np.asarray(image.shape).astype(np.int32)
        return image, image_shape

    def layer_op(self):
        image_to_sample, im_s = self.iterator.get_next()
        # TODO preprocessing layer modifying
        #      image shapes will not be supported
        # assuming the same shape across modalities, using the first
        im_s.set_shape((self.batch_size, self.spatial_rank + 1))
        image_shape = tf.unstack(im_s, axis=-1)
        # Four images concatenated at the batch_size dim
        # TODO resizing moving image to the fixed target
        image_to_sample.set_shape(
            [self.batch_size] + [None] * (self.spatial_rank + 1))

        # TODO affine data augmentation here
        if self.spatial_rank == 3:
            window_channels = np.prod(self.window_size[self.spatial_rank:]) * 4
            # TODO if no affine augmentation:
            img_spatial_shape = image_shape[:self.spatial_rank]
            win_spatial_shape = [tf.constant(dim) for dim in
                                 self.window_size[:self.spatial_rank]]

            # TODO shifts dtype should be int?
            batch_shift = [tf.random_uniform(
                               shape=(self.window_per_image, 1),
                               maxval=tf.to_float(img[0] - win - 1))
                           for win, img in
                           zip(win_spatial_shape, img_spatial_shape)]
            batch_shift = tf.concat(batch_shift, axis=1)

            affine_constraints = ((1.0, 0.0, 0.0, None),
                                  (0.0, 1.0, 0.0, None),
                                  (0.0, 0.0, 1.0, None))
            computed_grid = AffineGridWarperLayer(
                source_shape=(None, None, None),
                output_shape=self.window_size[:self.spatial_rank],
                constraints=affine_constraints)(batch_shift)
            resampler = ResamplerLayer(
                interpolation='linear', boundary='replicate')
            windows = resampler(image_to_sample, computed_grid)
            if self.window_per_image == self.batch_size:
                out_batch_size = self.window_per_image
            else:
                out_batch_size = self.window_per_image * self.batch_size
            out_shape = [out_batch_size] + \
                        list(self.window_size[:self.spatial_rank]) + \
                        [window_channels]
            windows.set_shape(out_shape)
        return windows

    # overriding input buffers
    def run_threads(self, session, *args, **argvs):
        # do nothing
        session.run(self.iterator.initializer)

    def close_all(self):
        # do nothing
        pass
