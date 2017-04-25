# -*- coding: utf-8 -*-
import os
import sys
import pickle
import nibabel
import numpy as np
from argparse import ArgumentParser
import misc as util
from medpy.filter import IntensityRangeStandardization

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')

parser = ArgumentParser(description='Train histogram references for a dataset')
parser.add_argument('data_folder',
                    help='path to the folder containing the data '
                         'to use for training')
parser.add_argument('model_saving_folder',
                    help='folder where to save '
                         'the histogram references learned')
parser.add_argument('--thr', default=0.955,
                    help='threshold used to separate '
                         'the foreground from the background')
parser.add_argument('--prefix', default='std_hist',
                    help='output .pkl file name prefix')
argument = parser.parse_args()
thr = float(argument.thr)
data_folder = argument.data_folder
target_folder = argument.model_saving_folder


if __name__ == '__main__':
    list_modality = util.list_modality(data_folder)
    print('Modalities found: %s' % list_modality)
    for modality in list_modality:
        list_files = [os.path.join(data_folder, f_n)
                      for f_n in os.listdir(data_folder)
                      if f_n.endswith(('.nii', '.nii.gz'))
                      and modality in f_n]
        list_images = []
        for file_path in list_files:
            image = nibabel.load(file_path).get_data()
            # Assure that min of image is 0
            image = image - np.min(image)
            #TODO add downsampling?
            foreground = image[image > np.mean(image)*thr]
            list_images.append(foreground)
        print('Train histogram reference for modality %s' % modality)
        irs = IntensityRangeStandardization()
        irs = irs.train(list_images)
        save_path = os.path.join(target_folder,
                                 '%s_%s.pkl' % (argument.prefix, modality))
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        with open(save_path, 'wb') as f:
            pickle.dump(irs, f)
        print 'file saved %s' % save_path

