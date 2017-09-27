
import numpy as np
import h5py
import os
import random


def random_transform_generator(batch_size, cornerScale=.1):
    offsets = np.tile([[[1.,1.,1.],[1.,1.,-1.],[1.,-1.,1.],[-1.,1.,1.]]],[batch_size,1,1])*np.random.uniform(0,cornerScale,[batch_size,4,3])
    newCorners = np.transpose(np.concatenate((np.tile([[[-1.,-1.,-1.],[-1.,-1.,1.],[-1.,1.,-1.],[1.,-1.,-1.]]],[batch_size,1,1])+offsets,np.ones([batch_size,4,1])),2),[0,1,2]) # O = T I
    srcCorners = np.tile(np.transpose([[[-1.,-1.,-1.,1.],[-1.,-1.,1.,1.],[-1.,1.,-1.,1.],[1.,-1.,-1.,1.]]],[0,1,2]),[batch_size,1,1])
    transforms = np.array([np.linalg.lstsq(srcCorners[k], newCorners[k])[0] for k in range(srcCorners.shape[0])])
    transforms = transforms*np.concatenate((np.ones([batch_size,1,2]),(-1)**np.random.randint(0,2,[batch_size,1,1]),np.ones([batch_size,1,1])),2) # random LR flip
    transforms = np.reshape(np.transpose(transforms[:][:,:][:,:,:3],[0,2,1]),[-1,1,12])
    return transforms


def initial_transform_generator(batch_size):
    identity = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]])
    identity = identity.flatten()
    transforms = np.reshape(np.tile(identity, batch_size), [batch_size, 1, 12])
    return transforms


def get_reference_grid(grid_size):
    return np.stack(np.meshgrid(np.arange(grid_size[0]), np.arange(grid_size[1]), np.arange(grid_size[2]), indexing='ij'), axis=3)


def dataset_switcher(dataset_name='test1-us'):

    if dataset_name == 'test-1-us1':
        label = 'Scratch/data/mrusv2/normalised/us_labels_sigma-1.h5'
        image = 'Scratch/data/mrusv2/normalised/us_images_vd1fov110.h5'
        mask = 'Scratch/data/mrusv2/normalised/us_masks_vd1fov110.h5'
        size = [65, 123, 76]
    elif dataset_name == 'test-1-mr1':
        label = 'Scratch/data/mrusv2/normalised/mr_labels_sigma-1.h5'
        image = 'Scratch/data/mrusv2/normalised/mr_images_vd1.h5'
        mask = ''
        size = [96, 128, 128]
    elif dataset_name == 'test3-us1':
        label = 'Scratch/data/mrusv2/normalised/us_labels_sigma3.h5'
        image = 'Scratch/data/mrusv2/normalised/us_images_vd1fov110.h5'
        mask = 'Scratch/data/mrusv2/normalised/us_masks_vd1fov110.h5'
        size = [65, 123, 76]
    elif dataset_name == 'test3-mr1':
        label = 'Scratch/data/mrusv2/normalised/mr_labels_sigma3.h5'
        image = 'Scratch/data/mrusv2/normalised/mr_images_vd1.h5'
        mask = ''
        size = [96, 128, 128]
    elif dataset_name == 'test1-us1':
        label = 'Scratch/data/mrusv2/normalised/us_labels_sigma1.h5'
        image = 'Scratch/data/mrusv2/normalised/us_images_vd1fov110.h5'
        mask = 'Scratch/data/mrusv2/normalised/us_masks_vd1fov110.h5'
        size = [65, 123, 76]
    elif dataset_name == 'test1-mr1':
        label = 'Scratch/data/mrusv2/normalised/mr_labels_sigma1.h5'
        image = 'Scratch/data/mrusv2/normalised/mr_images_vd1.h5'
        mask = ''
        size = [96, 128, 128]
    else:
        return

    label = os.path.join(os.environ['HOME'], label)
    image = os.path.join(os.environ['HOME'], image)
    if mask:
        mask = os.path.join(os.environ['HOME'], mask)
    dataset_size = len(h5py.File(image))
    return image, label, size, dataset_size, mask


class DataFeeder:
    def __init__(self, fn_image, fn_label, fn_mask=''):
        self.fn_image = fn_image
        self.id_image = h5py.File(self.fn_image, 'r')
        self.fn_label = fn_label
        self.id_label = h5py.File(self.fn_label, 'r')
        self.num_labels = self.id_label['/num_labels'][0]
        self.fn_mask = fn_mask
        if len(fn_mask) > 0:
            self.id_mask = h5py.File(self.fn_mask, 'r')
        else:
            self.id_mask = []

    def get_image_batch(self, case_indices):
        group_names = ['/case%06d' % i for i in case_indices]
        return np.expand_dims(np.concatenate([np.expand_dims(self.id_image[i], axis=0) for i in group_names], axis=0), axis=4)

    def get_label_batch(self, case_indices, label_indices=None):
        # for serialised data
        # if not label_indices:
        #     label_indices = [random.randrange(self.num_labels[i]) for i in case_indices]
        group_names = ['/case%06d_label%03d' % (i, j) for (i, j) in zip(case_indices, label_indices)]
        return np.expand_dims(np.concatenate([np.expand_dims(self.id_label[i], axis=0) for i in group_names], axis=0), axis=4)

    def get_mask_batch(self, case_indices):
        group_names = ['/case%06d' % i for i in case_indices]
        return np.expand_dims(np.concatenate([np.expand_dims(self.id_mask[i], axis=0) for i in group_names], axis=0), axis=4)
