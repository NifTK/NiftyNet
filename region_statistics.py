import os.path
import scipy
import nibabel as nib
import numpy as np
from region_properties import RegionProperties
from util import MorphologyOps
import util

MEASURES = ('centre of mass', 'volume', 'surface', 'surface volume ratio',
            'compactness', 'mean', 'weighted_mean', 'skewness',
            'kurtosis', 'min', 'max', 'std',
            'quantile_25', 'quantile_50', 'quantile_75')
OUTPUT_FORMAT = '{:4f}'

def run(param):
    name_for_file = '{}_{}'.format(
            os.path.split(param.data_image_dir)[1],
            os.path.split(param.seg_image_dir)[1])
    print(name_for_file)
    list_names = None
    if not param.name_seg == '""':
        list_names = param.name_seg
    elif not param.list_file == '""':
        text_file = open("param.list_file", "r")
        list_names = text_file.read().split(',')
    else:
        seg_names = util.list_nifti_files(param.seg_image_dir)
        img_names = util.list_nifti_files(param.data_image_dir)
    if list_names is not None:
        for name in list_names:
            img_names_temp, seg_names_temp = util.list_img_seg_by_fname(
                    name, param.data_image_dir, param.seg_image_dir)
            if len(seg_names_temp) != len(img_names_temp):
                seg_names_new = [item for item in seg_names_temp \
                        for i in xrange(len(img_names_temp))]
                img_names_new = [img_names_temp for i in xrange(len(seg_names_temp))]
            else:
                seg_names_new = seg_names_temp
                img_names_new = img_names_temp
            seg_names.append(seg_names_new)
            img_names.append(img_names_new)
            name_for_file = name_for_file + '_'.join(str(e) for e in list_names)

    # Different situations are handled:
    #   1 output file per pair of Seg and img file.
    #   Seg can be labels, probability in 3D or probability 4D or label 4D.
    #   img can be 3D or 4D.
    for S in seg_names:
        seg_name = os.path.join(param.seg_image_dir, S)
        SegTest = nib.load(seg_name).get_data()
        # assuming either 3D or 4D segmentation input
        SegTest = np.expand_dims(SegTest, -1) if SegTest.ndim == 3 else SegTest
        for D in img_names:
            img_name = os.path.join(param.data_image_dir, D)
            img = nib.load(img_name).get_data()

            # Create name of report file
            out_name = '{}_{}_{}.csv'.format(
                    os.path.basename(seg_name).rsplit(param.ext)[0],
                    os.path.basename(img_name).rsplit(param.ext)[0],
                    param.name_out)
            result_file = open(os.path.join(param.save_out_dir, out_name),'w+')
            print 'write {} to {}'.format(out_name, param.save_out_dir)
            # First line of report file to be set up according to form of img
            print >> result_file, 'Dim,Label' + \
                    RegionProperties(None, img, MEASURES).header_str()

            for d in np.arange(0, SegTest.shape[3]):
                SegImg = np.squeeze(SegTest[..., d])
                #if param.type_stats == "binary" and np.max(SegImg) > 1:
                #    SegImg[SegImg >= param.threshold] = 1
                #    SegImg[SegImg < param.threshold] = 0

                if np.max(SegImg) > 1:
                    type_str = "Labels"
                    threshold_steps = np.arange(1, np.max(SegImg))
                elif len(np.unique(SegImg)) > 2:
                    type_str = "Probabilities"
                    threshold_steps = np.arange(0, 1, float(param.step))
                elif len(np.unique(SegImg)) == 2:
                    type_str = "Binary"
                    SegImg = MorphologyOps(SegImg, 24).forground_component()
                    threshold_steps = np.arange(1, np.max(SegImg))
                else:
                    pass
                print type_str

                for i in threshold_steps:
                    if type_str == "Labels" or type_str == "Binary":
                        seg = (SegImg == i)
                    else:
                        seg = np.copy(SegImg)
                        seg[seg < i] = 0  # threshold prob.
                    if np.count_nonzero(seg) == 0: continue
                    roi_stats = RegionProperties(seg, img, MEASURES)
                    print >> result_file, '%d,%d'%(d, i) + \
                            roi_stats.to_string(OUTPUT_FORMAT)
            result_file.close()
