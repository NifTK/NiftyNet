from __future__ import absolute_import, print_function
import os.path
import numpy as np

import niftynet.utilities.csv_table as csv_table
import niftynet.utilities.misc_io as io
import niftynet.utilities.subject as subject
from niftynet.utilities.misc_common import MorphologyOps
from niftynet.evaluation.region_properties import RegionProperties

MEASURES = ('centre of mass', 'volume', 'surface', 'surface volume ratio',
            'compactness', 'mean', 'weighted_mean', 'skewness',
            'kurtosis', 'min', 'max', 'std',
            'quantile_25', 'quantile_50', 'quantile_75', 'asm', 'contrast',
            'correlation',
            'sumsquare',
            'sum_average', 'idifferentmomment', 'sumentropy', 'entropy',
            'differencevariance', 'sumvariance',
            'differenceentropy', 'imc1', 'imc2')
OUTPUT_FORMAT = '{:4f}'
OUTPUT_FILE_PREFIX = 'ROIStatistics'


def run(param, csv_dict):
    print('compute ROIs in {}'.format(param.img_dir))
    print('using {} as masks'.format(param.seg_dir))
    csv_loader = csv_table.CSVTable(csv_dict=csv_dict, allow_missing=False)
    names = [csv_loader._csv_table[m][0] for m in range(
        0, len(csv_loader._csv_table))]
    seg_names = [subject.MultiModalFileList(csv_loader._csv_table[m][1]) for
                 m in range(0, len(csv_loader._csv_table))]
    img_names = [subject.MultiModalFileList(csv_loader._csv_table[m][3]) for
                 m in range(0, len(csv_loader._csv_table))]
    # seg_names = util.list_files(param.seg_dir)
    # img_names = util.list_files(param.img_dir)
    out_name = ''
    if param.seg_type == 'unique':
        # output
        out_name = '{}_{}_{}_{}.csv'.format(
            OUTPUT_FILE_PREFIX,
            os.path.split(param.ref_dir)[1],
            os.path.split(param.seg_dir)[1],
            param.save_name)
        iteration = 0
        while os.path.exists(os.path.join(param.save_csv_dir, out_name)):
            iteration += 1
            out_name = '{}_{}_{}_{}_{}.csv'.format(
                OUTPUT_FILE_PREFIX,
                os.path.split(param.ref_dir)[1],
                os.path.split(param.seg_dir)[1],
                param.save_name, str(iteration))

        print("Writing {} to {}".format(out_name, param.save_csv_dir))

    """Different situations are handled:
       seg: (W,H,D) or (W,H,D,N_channels)
            each element is a probability (0 - 1) indicating foreground
            each element is a discrete class label
            each element is a binary label (TODO: double check this)
       img: (W,H,D) or (W,H,D,N_channels) """
    flag_not_unique = True
    for (seg_name, img_name, name) in zip(seg_names, img_names, names):
        seg = io.csv_cell_to_volume_5d(seg_name)
        img = io.csv_cell_to_volume_5d(img_name)
        if out_name == '':
            out_name = '{}_{}_{}.csv'.format(
                OUTPUT_FILE_PREFIX,
                name, param.save_name
                # os.path.basename(img_name).rsplit(param.ext)[0]
            )
        out_stream = open(os.path.join(param.save_csv_dir, out_name), 'a+')
        print('write: {}'.format(out_name))
        print('to folder: {}'.format(param.save_csv_dir))
        # a trivial RegionProperties obj to produce header_str
        header_str = RegionProperties(None, img, MEASURES).header_str()
        if flag_not_unique:
            out_stream.write('Mask, Data, Dim,Label' + header_str + '\n')
        # print >> out_stream, 'Dim,Label' + header_str
        if param.seg_type == 'unique':
            flag_not_unique = False
        for d in np.arange(0, seg.shape[3]):
            seg_d = seg[:, :, :, d, 0]
            threshold_steps = [0.5]
            if param.seg_type == "discrete" and np.max(seg_d) < 1.2:
                seg_d[seg_d >= param.threshold] = True
                seg_d[seg_d < param.threshold] = False

            if np.max(seg_d) > 1:
                type_str = "Labels"
                threshold_steps = np.unique(seg_d).tolist()
            elif len(np.unique(seg_d)) > 2:
                type_str = "Probabilities"
                threshold_steps = np.arange(0, 1, float(param.step))
            elif len(np.unique(seg_d)) == 2 and not param.seg_type == 'unique':
                type_str = "Binary"
                seg_d = MorphologyOps(seg_d, 24).foreground_component()
                threshold_steps = np.arange(1, seg_d[1])
            elif param.seg_type == 'unique':
                type_str = "Binary"
                threshold_steps = [1]
            else:
                pass

            for n, i in enumerate(threshold_steps):
                print('{} of {} thresholding steps'.format(
                    n, len(threshold_steps)))
                if type_str == "Labels" or type_str == "Binary":
                    seg_d_binary = np.where(seg_d == i, np.ones_like(seg_d),
                                            np.zeros_like(seg_d))
                else:
                    seg_d_binary = np.copy(seg_d[1])
                    seg_d_binary[seg_d[0] < i] = 0  # threshold prob.
                if np.count_nonzero(seg_d_binary) == 0:
                    print("Empty foreground in thresholded image")
                    continue
                roi_stats = RegionProperties(seg_d_binary, img, MEASURES)
                fixed_fields = '{},{},{},{}'.format(
                    seg_name.multi_mod_filenames[0][0],
                    img_name.multi_mod_filenames[0][0],
                    d, i)
                out_stream.write(fixed_fields + roi_stats.to_string(
                    OUTPUT_FORMAT) + '\n')
                # print >> out_stream, \
                # fixed_fields + roi_stats.to_string(OUTPUT_FORMAT)
        if not param.seg_type == 'unique1':
            out_stream.close()
