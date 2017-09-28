import os
import numpy as np
import nibabel as nib
from scipy import ndimage
files = os.listdir('gt')


def get_largest_two_component(img, prt = False, threshold = None):
    s = ndimage.generate_binary_structure(3,2) # iterate structure
    labeled_array, numpatches = ndimage.label(img,s) # labeling
    sizes = ndimage.sum(img,labeled_array,range(1,numpatches+1))
    sizes_list = [sizes[i] for i in range(len(sizes))]
    sizes_list.sort()
    #if(prt):
    #    print('component size', sizes_list, flush = True)
    if(len(sizes) == 1):
        return img
    else:
        if(threshold):
            out_img = np.zeros_like(img)
            for temp_size in sizes_list:
                if(temp_size > threshold):
                    temp_lab = np.where(sizes == temp_size)[0] + 1
                    temp_cmp = labeled_array == temp_lab
                    out_img = (out_img + temp_cmp) > 0
            return out_img
        else:
            max_size1 = sizes_list[-1]
            max_size2 = sizes_list[-2]
            max_label1 = np.where(sizes == max_size1)[0] + 1
            max_label2 = np.where(sizes == max_size2)[0] + 1
            component1 = labeled_array == max_label1
            component2 = labeled_array == max_label2
            #if(prt):
            #    print(max_size2, max_size1, max_size2/max_size1, flush = True)
            if(max_size2*10 > max_size1):
                component1 = (component1 + component2) > 0

            return component1

dice = []
for f in files:
    gt_file = 'gt/' + f
    img_gt = nib.load(gt_file).get_data() > 0

    pred_file_1 = 'pred_whole_tumor_axial/' + f[:-12] + '_niftynet_out.nii.gz'
    pred_file_2 = 'pred_whole_tumor_coronal/' + f[:-12] + '_niftynet_out.nii.gz'
    pred_file_3 = 'pred_whole_tumor_sagittal/' + f[:-12] + '_niftynet_out.nii.gz'

    img_pred_1 = nib.load(pred_file_1).get_data()[...,0,1]
    img_pred_2 = nib.load(pred_file_2).get_data()[...,0,1]
    img_pred_3 = nib.load(pred_file_3).get_data()[...,0,1]

    img_pred = (img_pred_1 + img_pred_2 + img_pred_3) / 3.0
    #img_pred = img_pred_1
    #img_pred = img_pred_2
    #img_pred = img_pred_3
    img_pred = img_pred > 0.5
    struct = ndimage.generate_binary_structure(3, 2)
    img_pred = ndimage.morphology.binary_closing(img_pred, structure = struct)
    img_pred = get_largest_two_component(img_pred, False, 2000)

    true_pos = np.float(np.sum(img_gt * img_pred))
    union = np.float(np.sum(img_gt) + np.sum(img_pred))
    d = true_pos * 2.0 / union
    print('%s: %s'%(f[:-12], d))
    dice.append(d)
print('%s images mean %s std %s'%(len(dice), np.mean(dice), np.std(dice)))


