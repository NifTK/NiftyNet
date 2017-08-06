# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.input_buffer import InputBatchQueueRunner
from niftynet.io.image_window import ImageWindow
from niftynet.io.image_window import N_SPATIAL
from niftynet.layer.base_layer import Layer


class UniformSampler(Layer, InputBatchQueueRunner):
    """
    This class generators samples by uniformly sampling each input volume
    currently 4D input is supported, Height x Width x Depth x Modality
    """

    def __init__(self, reader, data_param, samples_per_subject):
        # TODO: padding
        self.reader = reader

        batch_size = 3
        capacity = max(samples_per_subject * 2, batch_size * 2)
        InputBatchQueueRunner.__init__(self,
                                       batch_size=batch_size,
                                       capacity=capacity,
                                       shuffle=True)
        Layer.__init__(self, name='input_buffer')

        self.window = ImageWindow.from_user_spec(self.reader.input_sources,
                                                 self.reader.shapes,
                                                 self.reader.dtypes,
                                                 data_param,
                                                 samples_per_subject)
        self._create_queue_and_ops(self.window.placeholders_dict)
        tf.logging.info("initialised sampler output {}".format(
            self.window.shapes))

        # running test
        sess = tf.Session()
        _iter = 0
        for x in self():
            sess.run(self._enqueue_op, feed_dict=x)
            _iter += 1
            print('enqueue {}'.format(_iter))
            if _iter == 2:
                break
        out = sess.run(self._dequeue_op(2))
        print('dequeue')
        print(out['image'].shape)
        print(out['image_location'])
        import pdb; pdb.set_trace()

    def layer_op(self):
        while True:
            subject_id, data = self.reader()
            if not data:
                break
            image_sizes = {name: data[name].shape
                           for name in self.window.fields}
            coordinates = rand_spatial_coordinates(subject_id,
                                                   self.window.shapes,
                                                   image_sizes,
                                                   self.window.n_samples)
            #  initialise output dict
            output_dict = self.window.create_dict()
            # fill output dict with data
            for name in list(data):
                # fill output coordinates
                location_array = output_dict[
                    self.window.coordinates_placeholder(name)]
                location_array[...] = coordinates[name]
                # fill output window array
                image_array = output_dict[
                    self.window.data_placeholder(name)]
                for (i, location) in enumerate(location_array[:, 1:]):
                    x_, y_, z_, _x, _y, _z = location
                    image_array[i, ...] = data[name][x_:_x, y_:_y, z_:_z, ...]
            yield output_dict


def rand_spatial_coordinates(subject_id, win_sizes, img_sizes, n_samples):
    uniq_spatial_size = set([img_size[:N_SPATIAL]
                             for img_size in list(img_sizes.values())])
    if len(uniq_spatial_size) > 1:
        tf.logging.fatal("Don't know how to generate sampling "
                         "locations: Spatial dimensions of the "
                         "grouped input sources are not "
                         "consistent. {}".format(uniq_spatial_sizes))
        raise NotImplementedError
    uniq_spatial_size = uniq_spatial_size.pop()
    # find spatial window location based on the largest spatial window
    spatial_win_sizes = [win_size[:N_SPATIAL]
                         for win_size in list(win_sizes.values())]
    spatial_win_sizes = np.asarray(spatial_win_sizes, dtype=np.int32)
    max_spatial_win = np.max(spatial_win_sizes, axis=0)
    max_coords = np.zeros((n_samples, N_SPATIAL), dtype=np.int32)
    for i in range(0, N_SPATIAL):
        max_coords[:, i] = np.random.randint(
            0, max(uniq_spatial_size[i] - max_spatial_win[i], 1), n_samples)

    # adjust max spatial coordinates based on each spatial window size
    all_coordinates = {}
    for mod in list(win_sizes):
        win_size = win_sizes[mod][:N_SPATIAL]
        subject_id = np.ones((n_samples,), dtype=np.int32) * subject_id
        spatial_coords = np.zeros(
            (n_samples, N_SPATIAL * 2), dtype=np.int32)
        # shift starting coords of the window
        # so that smaller windows are centred within the large windows
        half_win_diff = np.floor((max_spatial_win - win_size) / 2.0)
        spatial_coords[:, :N_SPATIAL] = \
            max_coords[:, :N_SPATIAL] + half_win_diff[:N_SPATIAL]

        spatial_coords[:, N_SPATIAL:] = \
            spatial_coords[:, :N_SPATIAL] + win_size[:N_SPATIAL]
        # include the subject id
        spatial_coords = np.append(
            subject_id[:, None], spatial_coords, axis=1)
        all_coordinates[mod] = spatial_coords
    return all_coordinates


    # def __init__(self,
    #             patch,
    #             volume_loader,
    #             patch_per_volume=1,
    #             data_augmentation_methods=None,
    #             name="uniform_sampler"):

    #    super(UniformSampler, self).__init__(patch=patch, name=name)
    #    self.volume_loader = volume_loader
    #    self.patch_per_volume = patch_per_volume
    #    if data_augmentation_methods is None:
    #        self.data_augmentation_layers = []
    #    else:
    #        self.data_augmentation_layers = data_augmentation_methods

    # def layer_op(self, batch_size=1):
    #    """
    #     problems:
    #        check how many modalities available
    #        check the colon operator
    #        automatically handle mutlimodal by matching dims?
    #    """
    #    spatial_rank = self.patch.spatial_rank
    #    local_layers = [deepcopy(x) for x in self.data_augmentation_layers]
    #    patch = deepcopy(self.patch)
    #    while self.volume_loader.has_next:
    #        img, seg, weight_map, idx = self.volume_loader()

    #        # to make sure all volumetric data have the same spatial dims
    #        # and match volumetric data shapes to the patch definition
    #        # (the matched result will be either 3d or 4d)
    #        img.spatial_rank = spatial_rank

    #        img.data = io.match_volume_shape_to_patch_definition(
    #            img.data, patch)
    #        if img.data.ndim == 5:
    #            raise NotImplementedError
    #            # time series data are not supported
    #        if seg is not None:
    #            seg.spatial_rank = spatial_rank
    #            seg.data = io.match_volume_shape_to_patch_definition(
    #                seg.data, patch)
    #        if weight_map is not None:
    #            weight_map.spatial_rank = spatial_rank
    #            weight_map.data = io.match_volume_shape_to_patch_definition(
    #                weight_map.data, patch)

    #        # apply volume level augmentation
    #        for aug in local_layers:
    #            aug.randomise(spatial_rank=spatial_rank)
    #            img, seg, weight_map = aug(img), aug(seg), aug(weight_map)

    #        # generates random spatial coordinates
    #        locations = rand_spatial_coordinates(img.spatial_rank,
    #                                             img.data.shape,
    #                                             patch.image_size,
    #                                             self.patch_per_volume)
    #        for loc in locations:
    #            patch.set_data(idx, loc, img, seg, weight_map)
    #            yield patch
