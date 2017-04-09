# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from six.moves import range

import data_augmentation as dataug
import util
from preprocess import HistNormaliser


class VolumeSampler(object):
    def __init__(self,
                 patients, modalities,
                 batch_size, image_size, label_size,
<<<<<<< HEAD
                 volume_padding, sample_per_volume=0,dict_load={'path':['','',''],'prefix':['T1','T2','FLAIR'],
                                                                'suffix':['','',''],'ext':['.nii.gz']}
                 ,dict_preprocess={'rotation':1,'normalisation':1,'spatial_scaling':1},
                 dict_sampling={'comp_label_values':[],
                 'minimum_sampling_elements':[0], 'minimum_ratio':[0], 'min_numb_labels':1}):
=======
                 volume_padding, volume_hist_path, sample_per_volume=0,
                 dict_preprocess={'rotation':1,
                                  'normalisation':1,
                                  'spatial_scaling':1},
                 sample_opts={'comp_label_values':[0],
                              'minimum_sampling_elements':[0],
                              'minimum_ratio':[0.01],
                              'min_numb_labels':1}):
        self.patients = patients
        self.modalities = modalities
>>>>>>> f5adbeecfde3f6e42378bea8b006b1b63ba9fa15
        self.batch_size = batch_size
        self.image_size = image_size
        self.label_size = label_size
        self.padding = volume_padding
<<<<<<< HEAD
        self.f_names = f_names
        self.preprocessor = HistNormaliser()
        self.dict_sampling = dict_sampling
        self.dict_preprocess = dict_preprocess
        self.dict_load = dict_load
=======
>>>>>>> f5adbeecfde3f6e42378bea8b006b1b63ba9fa15

        self.sample_per_volume = sample_per_volume
        self.preprocessor = HistNormaliser(volume_hist_path)
        self.sample_opts = sample_opts
        self.dict_preprocess = dict_preprocess


    def training_samples_from(self, data_dir):
        # generate random cubic samples/segmentation maps with augmentations
        def sampler_iterator():
            print('New thread: Random samples from'\
                  ' {} modality, {} patients.'.format(
                self.modalities, len(self.patients)))
            self.adapt_sample_opts()
            while True:
<<<<<<< HEAD
                idx = np.random.randint(0, len(self.f_names))

                file_ = self.f_names[idx]
                print(file_)
                img, seg = util.load_file(img_name=img_path + '/' + file_,
                                          seg_name=seg_path + '/' + file_)
                # volume-level data augmentation
                if self.dict_preprocess['spatial_scaling'] ==1:
                    img, seg = dataug.rand_spatial_scaling(img, seg)
                if self.dict_preprocess['normalisation']==1:
=======
                idx = np.random.randint(0, len(self.patients))
                img, seg = util.load_file(data_dir,
                                          self.patients[idx],
                                          with_seg=True)
                if self.dict_preprocess['spatial_scaling'] ==1:
                    # volume-level data augmentation
                    img, seg = dataug.rand_spatial_scaling(img, seg)
                if self.dict_preprocess['normalisation']==1:
                    # intensity histogram normalisation (randomised)
>>>>>>> f5adbeecfde3f6e42378bea8b006b1b63ba9fa15
                    img = self.preprocessor.intensity_normalisation(img, True)
                # padding to alleviate volume level boundary effects
                if self.padding > 0:
                    img = [np.pad(img[:, :, :, mod_i], self.padding, 'minimum')
                           for mod_i in range(img.shape[-1])]
                    img = np.stack(img, axis=-1)
                    seg = np.pad(seg, self.padding, 'minimum')

                self.adapt_dict_sampling()

                # randomly sample windows from the volume
<<<<<<< HEAD
                if self.dict_sampling['minimum_ratio']==[0] and self.dict_sampling['minimum_sampling_elements']==[0] \
                        and self.dict_sampling['min_numb_labels']==1 and len(self.dict_sampling['comp_label_values'])==0:
                    xs, xe, ys, ye, zs, ze = dataug.rand_window_location_3d(
                        img.shape, self.image_size, self.sample_per_volume)
                else:
                    print('dealing with specific sampling ')
                    for t in range(self.sample_per_volume):
                        print ('doing ',t)
                        xs[t], xe[t], ys[t], ye[t], zs[t], ze[t] = self.strategic_sampling(img, seg)

                for t in range(self.sample_per_volume):
                    x_ = xs[t];
                    y_ = ys[t];
                    z_ = zs[t]
                    _x = xe[t];
                    _y = ye[t];
                    _z = ze[t]
                    if self.flag_preprocess['rotation']==1:
                        cuboid, label = dataug.rand_rotation_3d(
                        img[x_:_x, y_:_y, z_:_z],
                        seg[x_:_x, y_:_y, z_:_z])
                    else:
                        cuboid = img[x_:_x, y_:_y, z_:_z]
                        label = seg[x_:_x, y_:_y, z_:_z]
=======
                if self.sample_opts['minimum_ratio']==[0] \
                        and self.sample_opts['minimum_sampling_elements']==[0] \
                        and self.sample_opts['min_numb_labels']==1 \
                        and len(self.sample_opts['comp_label_values'])==0:
                    location = dataug.rand_window_location_3d(
                        img.shape, self.image_size, self.sample_per_volume)
                else:
                    print('dealing with specific sampling ')
                    location = []
                    for t in range(self.sample_per_volume):
                        #print ('doing ',t)
                        location.append(self.strategic_sampling(seg))
                    location = np.asarray(location)
                for t in range(self.sample_per_volume):
                    x_, _x, y_, _y, z_, _z = location[t]

                    cuboid = img[x_:_x, y_:_y, z_:_z, :]
                    label = seg[x_:_x, y_:_y, z_:_z]
                    if self.dict_preprocess['rotation']==1:
                        # TODO rotation should be applied before extracting a subwindow
                        # to avoid loss of information
                        cuboid, label = dataug.rand_rotation_3d(cuboid, label)
>>>>>>> f5adbeecfde3f6e42378bea8b006b1b63ba9fa15
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

<<<<<<< HEAD
    def adapt_dict_sampling(self):
        if 'comp_label_values' not in self.dict_sampling.keys():
            self.dict_sampling['comp_label_values']=[]
        if 'minimum_ratio' not in self.dict_sampling.keys():
            self.dict_sampling['minimum_ratio'] = [0]
        if 'minimum_sampling_elements' not in self.dict_sampling.keys():
            self.dict_sampling['minimum_sampling_elements'] = [0]
        if 'min_numb_labels' not in self.dict_sampling.keys():
            self.dict_sampling['min_numb_labels'] = 1
=======
    # Function to make the sample_opts consistent if some fields are missing
    def adapt_sample_opts(self):
        print('adaptation of sample_opts')
        if 'comp_label_values' not in self.sample_opts.keys():
            self.sample_opts['comp_label_values']=[]
        if 'minimum_ratio' not in self.sample_opts.keys():
            self.sample_opts['minimum_ratio'] = [0]
        if 'minimum_sampling_elements' not in self.sample_opts.keys():
            self.sample_opts['minimum_sampling_elements'] = [0]
        if 'min_numb_labels' not in self.sample_opts.keys():
            self.sample_opts['min_numb_labels'] = 1
        if 'flag_pad_effect' not in self.sample_opts.keys():
            self.sample_opts['flag_pad_effect'] = 0
>>>>>>> f5adbeecfde3f6e42378bea8b006b1b63ba9fa15
        return

    def adapt_dict_preprocess(self):
        if 'rotation' not in self.dict_preprocess.keys():
            self.dict_preprocess['rotation'] = 1
        if 'spatial_scaling' not in self.dict_preprocess.keys():
            self.dict_preprocess['spatial_scaling'] = 1
        if 'normalisation' not in self.dict_preprocess.keys():
            self.dict_preprocess['normalisation'] = 1
        return

<<<<<<< HEAD


    def strategic_sampling(self,img,seg):
        # get the number of labels
        uni, counts = np.unique(np.array(seg).flatten(),return_counts=True)
        Lcomp_values = 0
        if len(self.dict_sampling['comp_label_values'])==0:
            Lcomp_values = 0
        else:
            Lcomp_values = len(self.dict_sampling['comp_label_values'])

        if len(uni) < self.dict_sampling['min_numb_labels'] and self.min_numb_labels > Lcomp_values:
            min_nlab = len(uni)
        else:
            min_nlab = self.dict_sampling['min_numb_labels']
        flag_test = 0
        iter =0
        while flag_test == 0 and iter<200:
            flag_test = 1
            iter = iter+1
            xs, xe, ys, ye, zs, ze = dataug.rand_window_location_3d(
                img.shape, self.image_size, 1)
            seg_test = seg[xs:xe, ys:ye, zs:ze]
            uni_test, count_test = np.unique(np.array(seg_test).flatten(), return_counts=1)
            if Lcomp_values > 0:
                inter = [val for val in uni_test if val in self.dict_sampling['comp_label_values']]
=======
    def strategic_sampling(self, seg):
        # get the number of labels
        uni, counts = np.unique(np.array(seg).flatten(),return_counts=True)
        Lcomp_values = 0
        if len(self.sample_opts['comp_label_values'])==0:
            Lcomp_values=0
        else:
            Lcomp_values=len(self.sample_opts['comp_label_values'])

        if len(uni) < self.sample_opts['min_numb_labels']\
                and self.min_numb_labels > Lcomp_values:
            min_nlab = len(uni)
        else:
            min_nlab=self.sample_opts['min_numb_labels']

        flag_test = 0
        iter =0
        while flag_test==0 and iter<200:
            flag_test = 1
            iter = iter+1
            location = dataug.rand_window_location_3d(
                seg.shape, self.image_size, 1)
            xs = location[0,0]
            xe = location[0,1]
            ys = location[0,2]
            ye = location[0,3]
            zs = location[0,4]
            ze = location[0,5]
            if self.sample_opts['flag_pad_effect']:
                seg_test = seg[xs+self.padding:xe-self.padding,
                               ys+self.padding:ye-self.padding,
                               zs+self.padding:ze-self.padding]
            else:
                seg_test = seg[xs:xe, ys:ye, zs:ze]
            uni_test, count_test = np.unique(np.array(seg_test).flatten(),return_counts=1)
            if Lcomp_values > 0:
                inter = [val for val in uni_test if val in self.sample_opts['comp_label_values']]
>>>>>>> f5adbeecfde3f6e42378bea8b006b1b63ba9fa15
            flag_test = 1
            numb_add_toCheck = min_nlab
            new_values = uni_test.copy()
            numb_checked = 0
            if len(inter) < Lcomp_values:
                flag_test = 0
                continue
            elif len(uni_test)<min_nlab:
<<<<<<< HEAD
                flag_test = 0
                continue
            elif Lcomp_values>0:
                indices = []
                indices = np.arange(np.array(uni_test).shape[0])[np.in1d(inter,np.array(uni_test).flatten())]
=======
                flag_test =0
                continue
            elif Lcomp_values>0:
                indices = []
                indices = np.arange(np.array(uni_test).shape[0])[np.in1d(np.array(uni_test),inter)]
                indices_uni = np.arange(np.array(uni_test).shape[0])[np.in1d(np.array(uni_test), inter)]
>>>>>>> f5adbeecfde3f6e42378bea8b006b1b63ba9fa15
                if len(indices)==0:
                    indices = [-1]
                min_sample = [0]
                min_ratio = [0]
<<<<<<< HEAD
                if len(self.dict_sampling['minimum_sampling_elements'])!=0:
                    if len(self.dict_sampling['minimum_sampling_elements'])>np.max(indices) and len(list(set(indices) - set([-1])))>0:
                        min_sample = [self.dict_sampling['minimum_sampling_elements'][i] for i in indices]
                    else :
                        min_sample = [np.min(self.dict_sampling['minimum_sampling_elements']) for i in range(0,len(inter))]
=======
                if len(self.sample_opts['minimum_sampling_elements'])!=0:
                    if len(self.sample_opts['minimum_sampling_elements'])>=len(indices) and len(list(set(indices) - set([-1])))>0:
                        min_sample = [self.sample_opts['minimum_sampling_elements'][i] for i in range(len(indices))]
                    else :
                        min_sample = [np.min(self.sample_opts['minimum_sampling_elements']) for i in range(0,len(inter))]
>>>>>>> f5adbeecfde3f6e42378bea8b006b1b63ba9fa15
                    for i in range(0,len(inter)):
                        if np.count_nonzero(seg_test == inter[i])<min_sample[i]:
                            flag_test = 0
                            break
<<<<<<< HEAD
                if len(self.dict_sampling['minimum_sampling_elements'])!=0:
                    if len(self.dict_sampling['minimum_ratio'])>np.max(indices):
                        min_ratio = [self.dict_sampling['minimum_ratio'][i] for i in indices]
                    else:
                        min_ratio =[np.min(self.dict_sampling['minimum_ratio']) for i in range(len(inter))]
=======
                if len(self.sample_opts['minimum_sampling_elements'])!=0:
                    if len(self.sample_opts['minimum_ratio'])>=len(indices):
                        min_ratio = [self.sample_opts['minimum_ratio'][i] for i in range(len(indices))]
                    else:
                        min_ratio =[np.min(self.sample_opts['minimum_ratio']) for i in range(len(inter))]
>>>>>>> f5adbeecfde3f6e42378bea8b006b1b63ba9fa15
                    for i in range(0,len(inter)):
                        if np.count_nonzero(seg_test == inter[i])/np.size(seg_test) < min_ratio[i]:
                            flag_test = 0
                numb_add_toCheck = min_nlab - len(indices)
                new_values = [element for i, element in enumerate(uni_test) if i not in indices]
            if numb_add_toCheck > 0:
                for i in range(0,len(new_values)):
                    if np.count_nonzero(seg_test == new_values[i]) >= np.min(min_sample) and np.count_nonzero(seg_test == new_values[i])/np.size(seg_test) >= np.min(min_ratio):
                        numb_checked+=1
                if numb_checked < numb_add_toCheck:
                    flag_test = 0
<<<<<<< HEAD
        print('success after', iter)
        return xs,xe,ys,ye,zs,ze


    def grid_samples_from(self, img_path, seg_path, grid_size):
=======
        #print('success after', iter)
        return location[0]


    def grid_samples_from(self, data_dir, grid_size, yield_seg=False):
>>>>>>> f5adbeecfde3f6e42378bea8b006b1b63ba9fa15
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
