import os.path
import scipy
import nibabel as nib
import numpy as np
from region_properties import RegionProperties
from utilities.misc import MorphologyOps
import utilities.misc as util

MEASURES = ('centre of mass', 'volume', 'surface', 'surface volume ratio',
            'compactness', 'mean', 'weighted_mean', 'skewness',
            'kurtosis', 'min', 'max', 'std',
            'quantile_25', 'quantile_50', 'quantile_75')
OUTPUT_FORMAT = '{:4f}'
OUTPUT_FILE_PREFIX = 'ROIStatistics'

def run(param):
    print('compute ROIs in {}'.format(param.img_dir))
    print('using {} as masks'.format(param.seg_dir))
    seg_names = util.list_files(param.seg_dir)
    img_names = util.list_files(param.img_dir)

    """Different situations are handled:
       seg: (W,H,D) or (W,H,D,N_channels)
            each element is a probability (0 - 1) indicating foreground
            each element is a discrete class label
            each element is a binary label (TODO: double check this)
       img: (W,H,D) or (W,H,D,N_channels) """
    for seg_name in seg_names:
        seg = nib.load(os.path.join(param.seg_dir, seg_name))
        seg = seg.get_data()
        # assuming either 3D or 4D segmentation input
        assert((seg.ndim == 3) or (seg.ndim == 4))
        seg = np.expand_dims(seg, -1) if seg.ndim == 3 else seg
        for img_name in img_names:
            img = nib.load(os.path.join(param.img_dir, img_name))
            img = img.get_data()
            print img.shape
            # Create name of report file
            out_name = '{}_{}_{}.csv'.format(
                    OUTPUT_FILE_PREFIX,
                    os.path.basename(seg_name).rsplit(param.ext)[0],
                    os.path.basename(img_name).rsplit(param.ext)[0])
            out_stream = open(os.path.join(param.save_csv_dir, out_name),'w+')
            print('write: {}'.format(out_name))
            print('to folder: {}'.format(param.save_csv_dir))
            # a trivial RegionProperties obj to produce header_str
            header_str = RegionProperties(None, img, MEASURES).header_str()
            print >> out_stream, 'Dim,Label' + header_str

            for d in np.arange(0, seg.shape[3]):
                seg_d = np.squeeze(seg[..., d])
                if param.seg_type == "discrete" and np.max(seg_d) > 1:
                    seg_d[seg_d >= param.threshold] = True
                    seg_d[seg_d < param.threshold] = False

                if np.max(seg_d) > 1:
                    type_str = "Labels"
                    threshold_steps = np.unique(seg_d).tolist()
                elif len(np.unique(seg_d)) > 2:
                    type_str = "Probabilities"
                    threshold_steps = np.arange(0, 1, float(param.step))
                elif len(np.unique(seg_d)) == 2:
                    type_str = "Binary"
                    seg_d = MorphologyOps(seg_d, 24).foreground_component()
                    threshold_steps = np.arange(1, np.max(seg_d))
                else:
                    pass
                print type_str

                for n, i in enumerate(threshold_steps):
                    print('{} of {} thresholding steps'.format(
                        n, len(threshold_steps)))
                    if type_str == "Labels" or type_str == "Binary":
                        seg_d_binary = (seg_d == i)
                    else:
                        seg_d_binary = np.copy(seg_d)
                        seg_d_binary[seg_d < i] = 0  # threshold prob.
                    if np.count_nonzero(seg_d) == 0:
                        print("Empty foreground in thresholded image")
                        continue
                    roi_stats = RegionProperties(seg_d_binary, img, MEASURES)
                    fixed_fields = '{},{}'.format(d, i)
                    print >> out_stream, \
                            fixed_fields + roi_stats.to_string(OUTPUT_FORMAT)
            out_stream.close()
