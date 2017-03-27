import numpy as np
import util
import data_augmentation as DataAug
from preprocess import HistNormaliser


class VolumeSampler(object):
    def __init__(self, f_names,
                 batch_size, image_size, label_size,
                 volume_padding, sample_per_volume=0):
        self.batch_size = batch_size
        self.image_size = image_size
        self.label_size = label_size
        self.sample_per_volume = sample_per_volume
        self.padding = volume_padding
        self.f_names = f_names
        self.preprocessor = HistNormaliser()

    def training_samples_from(self, img_path, seg_path):
        # generate random cubic samples/segmentation maps with augmentations
        def sampler_iterator():
            print 'New threads: random samples from {} volumes.'.format(
                    len(self.f_names))
            while True:
                idx = np.random.randint(0, len(self.f_names))
                file_ = self.f_names[idx]
                img, seg = util.load_file(img_name=img_path + '/' + file_,
                                     seg_name=seg_path + '/' + file_)
                # volume-level data augmentation
                img, seg = DataAug.rand_spatial_scaling(img, seg)
                img = self.preprocessor.intensity_normalisation(img, True)
                # padding to alleviate volume level boundary effects
                if self.padding > 0:
                    img = np.pad(img, self.padding, 'minimum')
                    seg = np.pad(seg, self.padding, 'minimum')

                # randomly sample windows from the volume
                xs, xe, ys, ye, zs, ze = DataAug.rand_window_location_3d(
                    img.shape, self.image_size, self.sample_per_volume)
                for t in xrange(self.sample_per_volume):
                    x_ = xs[t]; y_ = ys[t]; z_ = zs[t]
                    _x = xe[t]; _y = ye[t]; _z = ze[t]
                    cuboid, label = DataAug.rand_rotation_3d(
                        img[x_:_x, y_:_y, z_:_z],
                        seg[x_:_x, y_:_y, z_:_z])
                    info = np.asarray(
                        [idx, x_, y_, z_, _x, _y, _z], dtype=np.int64)
                    #print '%s, %d'%(file_, t)
                    #print('sample from: %dx%dx%d'%(x_,y_,z_))
                    #print('sample to: %dx%dx%d'%(_x,_y,_z))
                    yield cuboid, label, info
        return sampler_iterator

    def grid_samples_from(self, img_path, seg_path, grid_size):
        # generate dense samples from a fixed sampling grid
        def sampler_iterator():
            for idx in xrange(len(self.f_names)):
                file_ = self.f_names[idx]
                img_name = img_path + '/' + file_
                seg_name = (seg_path + '/' + file_) if seg_path else None
                print '%d of %d loading %s'%(idx+1, len(self.f_names), img_name)
                img, seg = util.load_file(img_name, seg_name)
                img = self.preprocessor.intensity_normalisation(img)
                if self.padding > 0:
                    img = np.pad(img, self.padding, 'minimum')
                    seg = np.pad(seg, self.padding, 'minimum')\
                        if seg is not None else None
                xs, xe, ys, ye, zs, ze = DataAug.grid_window_location_3d(
                    img.shape, self.image_size, grid_size)
                n_windows = len(xs)
                print '{} samples of {}^3-voxels from {}-voxels volume'.format(
                    n_windows, self.image_size, img.shape)
                ids = np.array(range(n_windows))
                for j in xrange(n_windows + n_windows%self.batch_size):
                    i = ids[j % n_windows]
                    x_ = xs[i]; y_ = ys[i]; z_ = zs[i]
                    _x = xe[i]; _y = ye[i]; _z = ze[i]
                    cuboid = img[x_:_x, y_:_y, z_:_z]
                    info = np.asarray(
                        [idx, x_, y_, z_, _x, _y, _z], dtype=np.int64)
                    #print('grid sample from: %dx%dx%d to %dx%dx%d,'\
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
