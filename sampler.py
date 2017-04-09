import numpy as np
from six.moves import range

import data_augmentation as dataug
import util
from preprocess import HistNormaliser


class VolumeSampler(object):
    def __init__(self, f_names,
                 batch_size, image_size, label_size,
                 volume_padding, sample_per_volume=0,dict_load={'path':['','',''],'prefix':['T1','T2','FLAIR'],
                                                                'suffix':['','',''],'ext':['.nii.gz']}
                 ,dict_preprocess={'rotation':1,'normalisation':1,'spatial_scaling':1},
                 dict_sampling={'comp_label_values':[],
                 'minimum_sampling_elements':[0], 'minimum_ratio':[0], 'min_numb_labels':1}):
        self.batch_size = batch_size
        self.image_size = image_size
        self.label_size = label_size
        self.sample_per_volume = sample_per_volume
        self.padding = volume_padding
        self.f_names = f_names
        self.preprocessor = HistNormaliser()
        self.dict_sampling = dict_sampling
        self.dict_preprocess = dict_preprocess
        self.dict_load = dict_load

    def training_samples_from(self, img_path, seg_path):
        # generate random cubic samples/segmentation maps with augmentations
        def sampler_iterator():
            print('New threads: random samples from {} volumes.'.format(len(self.f_names)))
            while True:
                idx = np.random.randint(0, len(self.f_names))

                file_ = self.f_names[idx]
                print(file_)
                img, seg = util.load_file(img_name=img_path + '/' + file_,
                                          seg_name=seg_path + '/' + file_)
                # volume-level data augmentation
                if self.dict_preprocess['spatial_scaling'] ==1:
                    img, seg = dataug.rand_spatial_scaling(img, seg)
                if self.dict_preprocess['normalisation']==1:
                    img = self.preprocessor.intensity_normalisation(img, True)
                # padding to alleviate volume level boundary effects
                if self.padding > 0:
                    img = np.pad(img, self.padding, 'minimum')
                    seg = np.pad(seg, self.padding, 'minimum')

                self.adapt_dict_sampling()

                # randomly sample windows from the volume
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
                    info = np.asarray(
                        [idx, x_, y_, z_, _x, _y, _z], dtype=np.int64)
                    # print('%s, %d'%(file_, t))
                    # print('sample from: %dx%dx%d'%(x_,y_,z_))
                    # print('sample to: %dx%dx%d'%(_x,_y,_z))
                    yield cuboid, label, info

        return sampler_iterator

    def adapt_dict_sampling(self):
        if 'comp_label_values' not in self.dict_sampling.keys():
            self.dict_sampling['comp_label_values']=[]
        if 'minimum_ratio' not in self.dict_sampling.keys():
            self.dict_sampling['minimum_ratio'] = [0]
        if 'minimum_sampling_elements' not in self.dict_sampling.keys():
            self.dict_sampling['minimum_sampling_elements'] = [0]
        if 'min_numb_labels' not in self.dict_sampling.keys():
            self.dict_sampling['min_numb_labels'] = 1
        return

    def adapt_dict_preprocess(self):
        if 'rotation' not in self.dict_preprocess.keys():
            self.dict_preprocess['rotation'] = 1
        if 'spatial_scaling' not in self.dict_preprocess.keys():
            self.dict_preprocess['spatial_scaling'] = 1
        if 'normalisation' not in self.dict_preprocess.keys():
            self.dict_preprocess['normalisation'] = 1
        return



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
            flag_test = 1
            numb_add_toCheck = min_nlab
            new_values = uni_test.copy()
            numb_checked = 0
            if len(inter) < Lcomp_values:
                flag_test = 0
                continue
            elif len(uni_test)<min_nlab:
                flag_test = 0
                continue
            elif Lcomp_values>0:
                indices = []
                indices = np.arange(np.array(uni_test).shape[0])[np.in1d(inter,np.array(uni_test).flatten())]
                if len(indices)==0:
                    indices = [-1]
                min_sample = [0]
                min_ratio = [0]
                if len(self.dict_sampling['minimum_sampling_elements'])!=0:
                    if len(self.dict_sampling['minimum_sampling_elements'])>np.max(indices) and len(list(set(indices) - set([-1])))>0:
                        min_sample = [self.dict_sampling['minimum_sampling_elements'][i] for i in indices]
                    else :
                        min_sample = [np.min(self.dict_sampling['minimum_sampling_elements']) for i in range(0,len(inter))]
                    for i in range(0,len(inter)):
                        if np.count_nonzero(seg_test == inter[i])<min_sample[i]:
                            flag_test = 0
                            break
                if len(self.dict_sampling['minimum_sampling_elements'])!=0:
                    if len(self.dict_sampling['minimum_ratio'])>np.max(indices):
                        min_ratio = [self.dict_sampling['minimum_ratio'][i] for i in indices]
                    else:
                        min_ratio =[np.min(self.dict_sampling['minimum_ratio']) for i in range(len(inter))]
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
        print('success after', iter)
        return xs,xe,ys,ye,zs,ze


    def grid_samples_from(self, img_path, seg_path, grid_size):
        # generate dense samples from a fixed sampling grid
        def sampler_iterator():
            for idx in range(len(self.f_names)):
                file_ = self.f_names[idx]
                img_name = img_path + '/' + file_
                seg_name = (seg_path + '/' + file_) if seg_path else None
                print('%d of %d loading %s' % (idx + 1, len(self.f_names), img_name))
                img, seg = util.load_file(img_name, seg_name)
                img = self.preprocessor.intensity_normalisation(img)
                if self.padding > 0:
                    img = np.pad(img, self.padding, 'minimum')
                    seg = np.pad(seg, self.padding, 'minimum') \
                        if seg is not None else None
                xs, xe, ys, ye, zs, ze = dataug.grid_window_location_3d(
                    img.shape, self.image_size, grid_size)
                n_windows = len(xs)
                print('{} samples of {}^3-voxels from {}-voxels volume'.format(
                    n_windows, self.image_size, img.shape))
                ids = np.array(range(n_windows))
                for j in range(n_windows + n_windows % self.batch_size):
                    i = ids[j % n_windows]
                    x_ = xs[i];
                    y_ = ys[i];
                    z_ = zs[i]
                    _x = xe[i];
                    _y = ye[i];
                    _z = ze[i]
                    cuboid = img[x_:_x, y_:_y, z_:_z]
                    info = np.asarray(
                        [idx, x_, y_, z_, _x, _y, _z], dtype=np.int64)
                    # print('grid sample from: %dx%dx%d to %dx%dx%d,'\
                    #      'mean: %.4f, std: %.4f'%(info[1], info[2], info[3],
                    #                               info[4], info[5], info[6],
                    #                               np.mean(cuboid),
                    #                               np.std(cuboid)))
                    if seg is not None:
                        label = seg[x_:_x, y_:_y, z_:_z]
                        yield cuboid, label, info
                    else:
                        yield cuboid, info

        return sampler_iterator
