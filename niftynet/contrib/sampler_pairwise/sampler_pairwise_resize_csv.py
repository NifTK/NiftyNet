from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
#from tensorflow.contrib.data.python.ops.dataset_ops import Dataset

from niftynet.engine.image_window import ImageWindow
from niftynet.layer.base_layer import Layer
from niftynet.layer.grid_warper import AffineGridWarperLayer
from niftynet.layer.resampler import ResamplerLayer
from niftynet.layer.linear_resize import LinearResizeLayer as Resize


class PairwiseResizeSampler(Layer):
    def __init__(self,
                 reader_0,
                 reader_1,
                 data_param,
                 batch_size=1):
        Layer.__init__(self, name='pairwise_sampler_resize')
        # reader for the fixed images
        self.reader_0 = reader_0
        # reader for the moving images
        self.reader_1 = reader_1

        # TODO:
        # 0) check the readers should have the same length file list
        # 1) detect window shape mismatches or defaulting
        #    windows to the fixed image reader properties
        # 2) reshape images to (supporting multi-modal data)
        #    [batch, x, y, channel] or [batch, x, y, z, channels]
        # 3) infer spatial rank
        # 4) make ``label`` optional
        self.batch_size = int(batch_size)
        assert self.batch_size > 0, "batch size must be greater than 0"
        self.spatial_rank = 3
        self.window = ImageWindow.from_data_reader_properties(
            self.reader_0.input_sources,
            self.reader_0.shapes,
            self.reader_0.tf_dtypes,
            data_param)
        if self.window.has_dynamic_shapes:
            tf.logging.fatal('Dynamic shapes not supported.\nPlease specify '
                             'all spatial dims of the input data, for the '
                             'spatial_window_size parameter.')
            raise NotImplementedError
        # TODO: check spatial dims the same across input modalities
        self.image_shape = \
            self.reader_0.shapes['fixed_image'][:self.spatial_rank]
        self.moving_image_shape = \
            self.reader_1.shapes['moving_image'][:self.spatial_rank]
        self.window_size = self.window.shapes['fixed_image'][1:]

        # initialise a dataset prefetching pairs of image and label volumes
        n_subjects = len(self.reader_0.output_list)
        int_seq = list(range(n_subjects))
        # make the list of sequence divisible by batch size
        while len(int_seq) > 0 and len(int_seq) % self.batch_size != 0:
            int_seq.append(int_seq[-1])

        image_dataset = tf.data.Dataset.from_tensor_slices(int_seq)
        # mapping random integer id to 4 volumes moving/fixed x image/label
        # tf.py_func wrapper of ``get_pairwise_inputs``
        image_dataset = image_dataset.map(
            lambda image_id: tuple(tf.py_func(
                self.get_pairwise_inputs, [image_id],
                [tf.int32, tf.float32, tf.float32, tf.int32, tf.int32])),
            num_parallel_calls=4)  # supported by tf 1.4?
        # todo: sequential and no repeatition
        image_dataset = image_dataset.batch(self.batch_size)
        self.iterator = image_dataset.make_initializable_iterator()

    def get_pairwise_inputs(self, image_id):
        # fetch fixed image
        fixed_inputs = []
        fixed_inputs.append(self._get_image('fixed_image', image_id)[0])
        fixed_inputs.append(self._get_image('fixed_label', image_id)[0])
        fixed_inputs = np.concatenate(fixed_inputs, axis=-1)
        fixed_shape = np.asarray(fixed_inputs.shape).T.astype(np.int32)

        # fetch moving image
        moving_inputs = []
        moving_inputs.append(self._get_image('moving_image', image_id)[0])
        moving_inputs.append(self._get_image('moving_label', image_id)[0])
        moving_inputs = np.concatenate(moving_inputs, axis=-1)
        moving_shape = np.asarray(moving_inputs.shape).T.astype(np.int32)

        return image_id, fixed_inputs, moving_inputs, fixed_shape, moving_shape

    def _get_image(self, image_source_type, image_id):
        # returns a random image from either the list of fixed images
        # or the list of moving images
        try:
            image_source_type = image_source_type.decode()
        except AttributeError:
            pass
        if image_source_type.startswith('fixed'):
            _, data, _ = self.reader_0(idx=image_id)
        else:  # image_source_type.startswith('moving'):
            _, data, _ = self.reader_1(idx=image_id)
        image = np.asarray(data[image_source_type]).astype(np.float32)
        image_shape = list(image.shape)
        image = np.reshape(image, image_shape[:self.spatial_rank] + [-1])
        image_shape = np.asarray(image.shape).astype(np.int32)
        return image, image_shape

    def layer_op(self):
        image_id, fixed_inputs, moving_inputs, fixed_shape, moving_shape = \
            self.iterator.get_next()
        # TODO preprocessing layer modifying
        #      image shapes will not be supported
        # assuming the same shape across modalities, using the first
        image_id.set_shape((self.batch_size,))
        image_id = tf.to_float(image_id)

        fixed_inputs.set_shape(
            (self.batch_size,) + (None,) * self.spatial_rank + (2,))
        # last dim is 1 image + 1 label
        moving_inputs.set_shape(
            (self.batch_size,) + self.moving_image_shape + (2,))
        fixed_shape.set_shape((self.batch_size, self.spatial_rank + 1))
        moving_shape.set_shape((self.batch_size, self.spatial_rank + 1))

        # resizing the moving_inputs to match the target
        # assumes the same shape across the batch
        target_spatial_shape = \
            tf.unstack(fixed_shape[0], axis=0)[:self.spatial_rank]
        moving_inputs = Resize(new_size=target_spatial_shape)(moving_inputs)
        combined_volume = tf.concat([fixed_inputs, moving_inputs], axis=-1)

        # TODO affine data augmentation here
        if self.spatial_rank == 3:

            window_channels = np.prod(self.window_size[self.spatial_rank:]) * 4
            # TODO if no affine augmentation:
            img_spatial_shape = target_spatial_shape
            win_spatial_shape = [tf.constant(dim) for dim in
                                 self.window_size[:self.spatial_rank]]

            # scale the image to new space
            batch_scale = [
                tf.reshape(tf.to_float(img) / tf.to_float(win), (1,1))
                for (win, img) in zip(win_spatial_shape, img_spatial_shape)]
            batch_scale = tf.concat(batch_scale, axis=1)
            affine_constraints = ((None, 0.0, 0.0, 0.0),
                                  (0.0, None, 0.0, 0.0),
                                  (0.0, 0.0, None, 0.0))
            computed_grid = AffineGridWarperLayer(
                source_shape=(None, None, None),
                output_shape=self.window_size[:self.spatial_rank],
                constraints=affine_constraints)(batch_scale)
            computed_grid.set_shape((1,) +
                                    self.window_size[:self.spatial_rank] +
                                    (self.spatial_rank,))
            resampler = ResamplerLayer(
                interpolation='linear', boundary='replicate')
            windows = resampler(combined_volume, computed_grid)
            out_shape = [self.batch_size] + \
                        list(self.window_size[:self.spatial_rank]) + \
                        [window_channels]
            windows.set_shape(out_shape)

            image_id = tf.reshape(image_id, (self.batch_size, 1))
            start_location = tf.zeros((self.batch_size, self.spatial_rank))
            end_location = tf.constant(self.window_size[:self.spatial_rank])
            end_location = tf.reshape(end_location, (1, self.spatial_rank))
            end_location = tf.to_float(tf.tile(
                end_location, [self.batch_size, 1]))
            locations = tf.concat([
                image_id, start_location, end_location], axis=1)
        return windows, locations
        #return windows, [tf.reduce_max(computed_grid), batch_scale]

    # overriding input buffers
    def run_threads(self, session, *args, **argvs):
        """
        To be called at the beginning of running graph variables
        """
        session.run(self.iterator.initializer)
        return

    def close_all(self):
        # do nothing
        pass
