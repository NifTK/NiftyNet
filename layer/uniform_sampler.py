# -*- coding: utf-8 -*-
import numpy as np
from six.moves import range

from .base_sampler import BaseSampler
import engine.data_augmentation as dataug


class UniformSampler(BaseSampler):
    """
    This class generators samples by uniformly sampling each input volume
    currently 4D input is supported, Hight x Width x Depth x Modality
    """

    def __init__(self,
                 patch,
                 volume_loader,
                 patch_per_volume=1,
                 name="uniform_sampler"):
        super(UniformSampler, self).__init__(patch=patch, name=name)
        self.volume_loader = volume_loader
        self.patch_per_volume = patch_per_volume

        self.do_reorientation = True
        self.do_resampling = True
        self.do_normalisation = True
        self.do_whitening = True

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

        while self.volume_loader.has_next:
            img, seg, _, idx = self.volume_loader(self.do_reorientation,
                                                  self.do_resampling,
                                                  self.do_normalisation,
                                                  self.do_whitening)
            if img.ndim == 5:
                img = np.squeeze(img, axis=(4,))
            if seg.ndim == 5:
                seg = np.squeeze(seg, axis=(4,))
            location = dataug.rand_window_location_3d(
                    img.shape, self.patch.image_size, self.patch_per_volume)
            for t in range(self.patch_per_volume):
                if self.patch.spatial_rank == 3:
                    x_, _x, y_, _y, z_, _z = location[t]

                    self.patch.image = img[x_:_x, y_:_y, z_:_z, :]
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
                                z_b : (self.patch.label_size + z_b), :]

                    if self.patch.has_weight_maps:
                        border = self.patch.image_size - \
                                 self.patch.weight_map_size
                        x_b, y_b, z_b = (x_+border), (y_+border), (z_+border)
                        self.patch.weight_map = weight_map[
                                x_b : (self.patch.weight_map_size + x_b),
                                y_b : (self.patch.weight_map_size + y_b),
                                z_b : (self.patch.weight_map_size + z_b), :]
                    yield self.patch

                elif self.patch.spatial_rank == 2:
                    raise NotImplementedError
