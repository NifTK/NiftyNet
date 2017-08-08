from __future__ import absolute_import, print_function

import os.path

import nibabel as nib
import numpy as np

import niftynet.utilities.csv_table as csv_table
from niftynet.evaluation.pairwise_measures import PairwiseMeasuresRegression

MEASURES = (
    'mse','rmse','mae','r2'
)
# MEASURES_NEW = ('ref volume', 'reg volume', 'tp', 'fp', 'fn', 'outline_error',
#             'detection_error', 'dice')
OUTPUT_FORMAT = '{:4f}'
OUTPUT_FILE_PREFIX = 'PairwiseMeasureReg'


def run(param, csv_dict):
    # output
    out_name = '{}_{}_{}.csv'.format(
        OUTPUT_FILE_PREFIX,
        os.path.split(param.ref_dir)[1],
        os.path.split(param.seg_dir)[1])
    print("Writing {} to {}".format(out_name, param.save_csv_dir))

    # inputs
    csv_loader = csv_table.CSVTable(csv_dict=csv_dict, allow_missing=False)
    reg_names = [csv_loader._csv_table[m][1][0][0] for m in range(
        0, len(csv_loader._csv_table))]
    ref_names = [csv_loader._csv_table[m][2][0][0] for m in range(
        0, len(csv_loader._csv_table))]
    # reg_names = util.list_files(param.reg_dir, param.ext)
    # ref_names = util.list_files(param.ref_dir, param.ext)
    pair_list = list(zip(reg_names, ref_names))
    # TODO check reg_names ref_names matching
    # TODO do we evaluate all combinations?
    # import itertools
    # pair_list = list(itertools.product(reg_names, ref_names))
    print("List of references is {}".format(ref_names))
    print("List of regressions is {}".format(reg_names))

    # prepare a header for csv
    with open(os.path.join(param.save_csv_dir, out_name), 'w+') as out_stream:
        # a trivial PairwiseMeasures obj to produce header_str
        m_headers = PairwiseMeasuresRegression(0, 0, measures=MEASURES).header_str()
        out_stream.write("Name (ref), Name (reg)" + m_headers + '\n')

        # do the pairwise evaluations
        for i, pair_ in enumerate(pair_list):
            reg_name = pair_[0]
            ref_name = pair_[1]
            print('>>> {} of {} evaluations, comparing {} and {}.'.format(
                i + 1, len(pair_list), ref_name, reg_name))
            reg_nii = nib.load(os.path.join(param.seg_dir, reg_name))
            ref_nii = nib.load(os.path.join(param.ref_dir, ref_name))
            voxel_sizes = reg_nii.header.get_zooms()[0:3]
            reg = np.squeeze(reg_nii.get_data())
            ref = np.squeeze(ref_nii.get_data())
            assert (np.all(reg) >= 0)
            assert (np.all(ref) >= 0)
            assert (reg.shape == ref.shape)
            PE = PairwiseMeasuresRegression(reg, ref,
                                  measures=MEASURES)
            fixed_fields = "{}, {}, ".format(ref_name, reg_name)
            out_stream.write(fixed_fields + PE.to_string(
                OUTPUT_FORMAT) + '\n')

