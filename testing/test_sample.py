import sys
import tensorflow as tf
from sampler import VolumeSampler

if __name__ == "__main__":
    img_folder = "/home/ucl/eisuc269/tnet/DataBrain/T1/"
    seg_folder = "/home/ucl/eisuc269/tnet/DataBrain/Lab_normalised/"

    rs = VolumeSampler(train_names, 1, 96, 96, 2, 20)
    f = rs.training_samples_from(img_folder, seg_folder)
    for img,lab,info in f():
        print info
        if info[-1] < 64:
            print info
        print img.shape
        print lab.shape
        #import pdb;pdb.set_trace()
