# -*- coding: utf-8 -*-
import numpy as np
from six.moves import range

import data_augmentation as dataug
import utilities.misc as util
from preprocess import HistNormaliser


class VolumeSampler(object):
    def __init__(self,
                 patients, modalities,
                 batch_size, image_size, label_size,
                 volume_padding, volume_hist_path, sample_per_volume=0,
                 dict_preprocess={'rotation': 1,
                                  'normalisation': 1,
                                  'spatial_scaling': 1},
                 sample_opts={'compulsory_labels': [[0], [0]],
                              'minimum_ratio': 0.01,
                              'min_numb_labels': 1,
                              'flag_pad_effect': 1},
                 ):
        self.patients = patients
        self.modalities = modalities
        self.batch_size = batch_size
        self.image_size = image_size
        self.label_size = label_size
        self.padding = volume_padding
        self.sample_per_volume = sample_per_volume
        self.preprocessor = HistNormaliser(volume_hist_path)
        self.sample_opts = sample_opts
        self.dict_preprocess = dict_preprocess

    # Function to make the sample_opts consistent if some fields are missing
    def adapt_sample_opts(self):
        print('adaptation of sample_opts')
        if 'compulsory_labels' not in self.sample_opts.keys():
            self.sample_opts['compulsory_labels']=[[], []]
        if 'minimum_ratio' not in self.sample_opts.keys():
            self.sample_opts['minimum_ratio'] = 0
        if 'min_numb_labels' not in self.sample_opts.keys():
            self.sample_opts['min_numb_labels'] = 1
        if 'flag_pad_effect' not in self.sample_opts.keys():
            self.sample_opts['flag_pad_effect'] = 0

    def uniform_sampling_from(self, data_dir):
        def sampler_iterator():
            print('New thread: uniform random samples from'\
                    ' {} modality, {} patients.'.format(
                        self.modalities, len(self.patients)))
            while True:
                idx = np.random.randint(0, len(self.patients))
                img, seg = util.load_file(
                        data_dir, self.patients[idx], with_seg=True)
                if self.dict_preprocess['spatial_scaling'] == 1:
                    # volume-level data augmentation
                    img, seg = dataug.rand_spatial_scaling(img, seg)
                if self.dict_preprocess['normalisation'] == 1:
                    # intensity histogram normalisation (randomised)
                    img = self.preprocessor.intensity_normalisation(img, True)
                if self.padding > 0:
                    # padding to alleviate volume level boundary effects
                    img = [np.pad(img[:, :, :, mod_i], self.padding, 'minimum')
                           for mod_i in range(img.shape[-1])]
                    img = np.stack(img, axis=-1)
                    seg = np.pad(seg, self.padding, 'minimum')
                if self.dict_preprocess['rotation']==1:
                    # volume-level randomised rotation
                    img, seg = dataug.rand_rotation_3d(img, seg)
                # randomly sample windows from the volume
                location = dataug.rand_window_location_3d(
                    img.shape, self.image_size, self.sample_per_volume)
                for t in range(self.sample_per_volume):
                    x_, _x, y_, _y, z_, _z = location[t]
                    cuboid = img[x_:_x, y_:_y, z_:_z, :]
                    label = seg[x_:_x, y_:_y, z_:_z]
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

    def grid_sampling_from(self, data_dir, grid_size, yield_seg=False):
        # generate dense samples from a fixed sampling grid
        def sampler_iterator():
            for idx in range(len(self.patients)):
                print('{} of {} loading {}'.format(
                    idx + 1, len(self.patients), self.patients[idx]))
                img, seg = util.load_file(
                        data_dir, self.patients[idx], with_seg=yield_seg)
                if self.dict_preprocess['normalisation'] == 1:
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
                            border = np.int(
                                    (self.image_size - self.label_size) / 2)
                            label = label[border : (self.label_size + border),
                                          border : (self.label_size + border),
                                          border : (self.label_size + border)]
                        yield cuboid, label, info
                    else:
                        yield cuboid, info
        return sampler_iterator

    def non_uniform_sampling_from(self, data_dir):
        # generate random cubic samples/segmentation maps with augmentations
        def sampler_iterator():
            print('New thread: Random samples from'\
                    ' {} modality, {} patients.'.format(
                      self.modalities, len(self.patients)))
            self.adapt_sample_opts()
            while True:
                idx = np.random.randint(0, len(self.patients))
                img, seg = util.load_file(data_dir,
                                          self.patients[idx],
                                          with_seg=True)
                if self.dict_preprocess['spatial_scaling'] ==1:
                    # volume-level data augmentation
                    img, seg = dataug.rand_spatial_scaling(img, seg)
                if self.dict_preprocess['normalisation'] == 1:
                    # intensity histogram normalisation (randomised)
                    img = self.preprocessor.intensity_normalisation(img, True)
                if self.padding > 0:
                    # padding to alleviate volume level boundary effects
                    img = [np.pad(img[:, :, :, mod_i], self.padding, 'minimum')
                           for mod_i in range(img.shape[-1])]
                    img = np.stack(img, axis=-1)
                    seg = np.pad(seg, self.padding, 'minimum')
                if self.dict_preprocess['rotation']==1:
                    img, seg = dataug.rand_rotation_3d(img, seg)

                # randomly sample windows from the volume
                print('dealing with specific sampling ')
                location = [self.strategic_sampling(seg)\
                        for t in range(self.sample_per_volume)]
                location = np.asarray(location)
                for t in range(self.sample_per_volume):
                    x_, _x, y_, _y, z_, _z = location[t]

                    cuboid = img[x_:_x, y_:_y, z_:_z, :]
                    label = seg[x_:_x, y_:_y, z_:_z]
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

    def strategic_sampling(self, seg):
        # get the number of labels
        uni, counts = np.unique(np.array(seg).flatten(), return_counts=True)
        #print(uni, counts)
        compulsory_values = self.sample_opts['compulsory_labels'][0]
        compulsory_ratios = self.sample_opts['compulsory_labels'][1]
        min_nlabels = self.sample_opts['min_numb_labels']
        min_ratio = self.sample_opts['minimum_ratio']
        # Check if there are enough different labels compared to existing
        # number of labels in seg
        if len(uni) < self.sample_opts['min_numb_labels']:
            flag_test = 0
            print (" not enough labels in seg")
            # The selected file does not have the minimum expected number of
            # labels. A random window is returned anyway
            return dataug.rand_window_location_3d(
                seg.shape, self.image_size, 1)
        # Check if all compulsory values are there
        flag_comp = 1
        for x in compulsory_values:
            if x not in uni:
                flag_comp = 0
        if not flag_comp:
            print("compulsory data not obtained")
            # The compulsory values are not all present in the image. A
            # random window is returned anyway...
            return dataug.rand_window_location_3d(
                seg.shape, self.image_size, 1)

        flag_test = 0
        iter = 0
        # Number of compulsory labels to find
        numb_compulsory = len(compulsory_values)
        while flag_test == 0 and iter < 200:
            flag_test = 1
            iter += 1
            location = dataug.rand_window_location_3d(
                seg.shape, self.image_size, 1)
            xs = location[0, 0]
            xe = location[0, 1]
            ys = location[0, 2]
            ye = location[0, 3]
            zs = location[0, 4]
            ze = location[0, 5]
            if self.sample_opts['flag_pad_effect']:
                seg_test = seg[xs + self.padding:xe - self.padding,
                           ys + self.padding:ye - self.padding,
                           zs + self.padding:ze - self.padding]
            else:
                seg_test = seg[xs:xe, ys:ye, zs:ze]
            uni_test, count_test = np.unique(np.array(seg_test).flatten(),
                                             return_counts=1)

            if numb_compulsory > 0:
                # Check if all the compulsory values are there and with the
                # adequate minimum ratio
                uni_list = uni_test.tolist()
                print(uni_list)
                for (x, r) in zip(compulsory_values, compulsory_ratios):
                    # If value is present, calculate ratio and compare to
                    # value expected

                    if x in uni_list:
                        rat_test = np.true_divide(count_test[uni_list.index(x)],
                                          np.sum(
                            count_test))
                        if rat_test < r:
                            flag_test = 0
                    if not x in uni_list:
                        flag_test = 0

            # Continue to check if other conditions are valid only if valid
                        # beforehand
            if flag_test:
                # Check if there are other labels to have with no value
                # compulsory and with a minimum ratio different from 0
                if min_nlabels > numb_compulsory and min_ratio > 0:
                    # Only things to check if number of different labels
                # required is higher than number of compulsory labels
                    numb_add_tocheck = min_nlabels - numb_compulsory
                    if len(uni_test) - numb_compulsory < numb_add_tocheck:
                        # Case where there are not enough additional labels
                        # once the compulsory ones have been found
                        flag_test = 0
                    else:
                        # Check all values in uni_test and corresponding
                        # ratio and see if the condition of number of added values to
                        # check is verified
                        numb_verified = 0
                        for i in xrange(0,len(uni_test)):
                            if uni_test[i] not in compulsory_values:
                                ratio = np.true_divide(count_test[i], np.sum(
                                    count_test))
                                if ratio > min_ratio:
                                    numb_verified += 1
                        if numb_verified < numb_add_tocheck:
                            flag_test = 0

        print('success after', iter)
        return location[0]
