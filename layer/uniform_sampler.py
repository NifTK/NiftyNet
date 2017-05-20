# -*- coding: utf-8 -*-
import numpy as np
from six.moves import range

from .base_sampler import BaseSampler
import nn.data_augmentation as dataug


class UniformSampler(BaseSampler):
    """
    This class generators samples by uniformly sampling each input volume
    """

    def __init__(self,
                 patch,
                 volume_loader,
                 patch_per_volume=1,
                 name="uniform_sampler"):
        super(UniformSampler, self).__init__(patch=patch, name=name)
        self.volume_loader = volume_loader
        self.patch_per_volume = patch_per_volume

    def layer_op(self, batch_size=1):
        """
         problems:
            self.volume_loader.has_next()
            sample_per_subject
            check sampler output dimensionalities
            check how many modalities available
            now assumes patch.label_size <= patch.image_size
            check the colon operator
            automatically handle mutlimodal by matching dims?
            assume img, seg the same size
        """
        # batch_size is needed here so that it generates total number of
        # N samples where (N % batch_size) == 0


        #while self.volume_loader.has_next():
        #    img, seg, _, patient = self.volume_loader.next_subject()
        #    for i <= param.sampler_per_volume:
        #        loc = search_random_location(img)
        #        self.patch.image = img[loc].astype(np.float32)
        #        self.patch.info = loc
        #        self.patch.label = seg[loc].astype(np.int64)
        #        yield self.patch

        #while True:
        for i in range(2 + len(self.volume_loader.subject_list)):
            idx, img, seg, weight_map = self.volume_loader.next_subject()
            location = dataug.rand_window_location_3d(
                    img.shape, self.patch.image_size, self.patch_per_volume)
            for t in range(self.patch_per_volume):
                if self.patch.spatial_rank == 3:
                    x_, _x, y_, _y, z_, _z = location[t]

                    self.patch.image = img[x_:_x, y_:_y, z_:_z]
                    self.patch.info = np.array([idx, x_, y_, z_, _x, _y, _z],
                                               dtype=np.int64)
                    if self.patch.has_labels:
                        border = self.patch.image_size - \
                                 self.patch.label_size
                        assert border >= 0 # assumes label_size <= image_size
                        x_b, y_b, z_b = (x_+border), (y_+border), (z_+border)
                        self.patch.label = seg[
                                x_b : (self.patch.label_size + x_b),
                                y_b : (self.patch.label_size + y_b),
                                z_b : (self.patch.label_size + z_b)]

                    if self.patch.has_weight_maps:
                        border = self.patch.image_size - \
                                 self.patch.weight_map_size
                        x_b, y_b, z_b = (x_+border), (y_+border), (z_+border)
                        self.patch.weight_map = weight_map[
                                x_b : (self.patch.weight_map_size + x_b),
                                y_b : (self.patch.weight_map_size + y_b),
                                z_b : (self.patch.weight_map_size + z_b)]
                    yield self.patch

                elif self.patch.spatial_rank == 2:
                    raise NotImplementedError
