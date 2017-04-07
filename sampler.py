# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from six.moves import range

import data_augmentation as dataug
import util
from preprocess import HistNormaliser


class VolumeSampler(object):
    def __init__(self, patients, modalities,
                 batch_size, image_size, label_size,
                 volume_padding, volume_hist_path, sample_per_volume=10):
        self.patients = patients
        self.modalities = modalities

        self.batch_size = batch_size
        self.image_size = image_size
        self.label_size = label_size

        self.padding = volume_padding
        self.sample_per_volume = sample_per_volume
        self.preprocessor = HistNormaliser(volume_hist_path)

    def training_samples_from(self, data_dir):
        # generate random cubic samples/segmentation maps with augmentations
        def sampler_iterator():
            print('New thread: Random samples from'\
                  ' {} modality, {} patients.'.format(
                self.modalities, len(self.patients)))
            while True:
                idx = np.random.randint(0, len(self.patients))
                img, seg = util.load_file(data_dir,
                                          self.patients[idx],
                                          with_seg=True)
                # volume-level data augmentation
                img, seg = dataug.rand_spatial_scaling(img, seg)
                img = self.preprocessor.intensity_normalisation(img, True)
                # padding to alleviate volume level boundary effects
                if self.padding > 0:
                    img =[np.pad(img[:, :, :, mod_i], self.padding, 'minimum')
                          for mod_i in range(img.shape[-1])]
                    img = np.stack(img, axis=-1)
                    seg = np.pad(seg, self.padding, 'minimum')

                # randomly sample windows from the volume
                location = dataug.rand_window_location_3d(
                    img.shape, self.image_size, self.sample_per_volume)
                for t in range(self.sample_per_volume):
                    x_, _x, y_, _y, z_, _z = location[t]
                    #xs[t], ys[t], zs[t], xe[t], ye[t], ze[t]
                    # TODO rotation should be applied before extracting a subwindow
                    # to avoid loss of information
                    cuboid, label = dataug.rand_rotation_3d(
                        img[x_:_x, y_:_y, z_:_z, :],
                        seg[x_:_x, y_:_y, z_:_z])
                    info = np.asarray(
                        [idx, x_, y_, z_, _x, _y, _z], dtype=np.int64)
                    if self.label_size < self.image_size:
                        border = np.int((self.image_size - self.label_size) / 2)
                        label = label[border : (self.label_size + border),
                                      border : (self.label_size + border),
                                      border : (self.label_size + border)]
                    #print('%s, %d'%(file_, t))
                    #print('sample from: %dx%dx%d'%(x_,y_,z_))
                    #print('sample to: %dx%dx%d'%(_x,_y,_z))
                    yield cuboid, label, info
        return sampler_iterator

    def grid_samples_from(self, data_dir, grid_size, yield_seg=False):
        # generate dense samples from a fixed sampling grid
        def sampler_iterator():
            for idx in range(len(self.patients)):
                print('{} of {} loading {}'.format(
                    idx + 1, len(self.patients), self.patients[idx]))
                img, seg = util.load_file(data_dir,
                                          self.patients[idx],
                                          with_seg=yield_seg)
                img = self.preprocessor.intensity_normalisation(img)
                if self.padding > 0:
                    img = [np.pad(img[:,:,:,mod_i], self.padding, 'minimum')
                           for mod_i in range(img.shape[-1])]
                    img = np.stack(img, axis=-1)
                    seg = np.pad(seg, self.padding, 'minimum') \
                        if (seg is not None) else None
                location = dataug.grid_window_location_3d(
                    img.shape[:-1], self.image_size, grid_size)
                n_windows = location.shape[0]
                print('{} samples of {}^3-voxels from {}-voxels volume'.format(
                    n_windows, self.image_size, img.shape))
                ids = np.array(range(n_windows))
                for j in range(n_windows + n_windows % self.batch_size):
                    i = ids[j % n_windows]
                    x_, _x, y_, _y, z_, _z = location[i]
                    cuboid = img[x_:_x, y_:_y, z_:_z, :]
                    info = np.asarray(
                        [idx, x_, y_, z_, _x, _y, _z], dtype=np.int64)
                    #print('grid sample from: %dx%dx%d to %dx%dx%d,'\
                    #       'mean: %.4f, std: %.4f'%(info[1], info[2], info[3],
                    #                                info[4], info[5], info[6],
                    #                                np.mean(cuboid),
                    #                                np.std(cuboid)))
                    if seg is not None:
                        label = seg[x_:_x, y_:_y, z_:_z]
                        if self.label_size < self.image_size:
                            border = np.int((self.image_size - self.label_size) / 2)
                            label = label[border : (self.label_size + border),
                                          border : (self.label_size + border),
                                          border : (self.label_size + border)]
                        yield cuboid, label, info
                    else:
                        yield cuboid, info
        return sampler_iterator
