import os
import time

import numpy as np
import tensorflow as tf


import util
from loss import LossFunction
from sampler import VolumeSampler
from nn_queue import TrainEvalInputBuffer
import data_augmentation

dict_load={'path':['/Users/csudre/Documents/Barcelona/BarcelonaT1Space/','/Users/csudre/Documents/Barcelona/BarcelonaT1Space/T2_T1Space',
                   '/Users/csudre/Documents/Barcelona/BarcelonaT1Space/Parcellations'], 'prefix':['T1_', 'T2_', ''],
           'suffix':['', '', '_003_SAG_FSPGR_3D_NeuroMorph_Parcellation']}

img_test = util.load_4d_img_from_dict(dict_load,'10015')

seg = np.squeeze(img_test[:,:,:,2])
mask = np.copy(seg)
mask[seg<80] = 1
mask[seg>90] = 1
mask[mask>2] = 0
from preprocess import  MahalNormaliser

img_test2 = np.squeeze(img_test[:,:,:,:-1])
mn = MahalNormaliser(mask,0.05)
img_res = mn.mahalanobis_normalisation(img_test2)
img_biased = data_augmentation.apply_rand_biasfield(img_res,max_range=0.5)

vs = VolumeSampler( ['10015_003_SAG_FSPGR_3D_NeuroMorph_Parcellation.nii.gz','10016_003_SAG_FSPGR_3D_NeuroMorph_Parcellation.nii.gz','10019_003_SAG_FSPGR_3D_NeuroMorph_Parcellation.nii.gz'],
                 1, 85, 85,
                 15, sample_per_volume=0,dict_preprocess={'rotation':1,'normalisation':1,'spatial_scaling':1},
                 dict_sampling={'comp_label_values':[52,65],
                 'minimum_sampling_elements':[10,20], 'minimum_ratio':[0], 'min_numb_labels':4})
idx = np.random.randint(0, len(vs.f_names))

file_ = vs.f_names[idx]
print(file_)
img, seg = util.load_file(img_name='/Users/csudre/Documents/Barcelona/BarcelonaT1Space/Parcellations' + '/' + file_,
                          seg_name='/Users/csudre/Documents/Barcelona/BarcelonaT1Space/Parcellations' + '/' + file_)
xs,xe,ys,ye,zs,ze=vs.strategic_sampling(img,seg)
# sample_generator = vs.training_samples_from(
#     ['/Users/csudre/Documents/Barcelona/BarcelonaT1Space/Parcellations'], '/Users/csudre/Documents/Barcelona/BarcelonaT1Space/Parcellations')
# train_batch_runner = TrainEvalInputBuffer(
#     1,
#     1,
#     shapes=[[85] * 3, [85] * 3, [7]],
#     sample_generator=sample_generator,
#     shuffle=True)
#
# train_pairs = train_batch_runner.pop_batch()


# labels = tf.constant([1,1,1,2,2,1,1,1,2,2,2,3,3,1,1,1],dtype=tf.int64)
# pred = tf.random_uniform([16,3],0,1)
# predFin = tf.div(pred,tf.tile(tf.expand_dims(tf.reduce_sum(pred,1),1),[1,3]))
# labelsFin = labels-1
# loss_func = LossFunction(3,'GDSC',reg_type=None,decay=-1)
# loss_func2 = LossFunction(3,'SensSpec',reg_type=None,decay=-1)

sess = tf.Session()
with sess as default:
    print(xs,xe,ys,ye,zs,ze)
    # img_pairs = train_pairs['images'].eval()
    # lab_pairs = train_pairs['labels'].eval()
    # info_pairs = train_pairs['info'].eval()
    with tf.name_scope('TestLoss') as scope:
        GDSC = loss_func.total_loss(predFin, labelsFin,scope)
        SensSpec = loss_func2.total_loss(predFin,labelsFin,scope)
        print(GDSC.eval(),SensSpec.eval())