import os
import os.path
import sys
import time
import scipy
import nibabel
import numpy as np
import tensorflow as tf
from skimage import measure
from sklearn.neighbors import DistanceMetric
from scipy import ndimage

class get_morphology(object):
    def __init__(self, seg_img,neigh):
        self.seg = seg_img
        self.neigh = neigh

    def BorderMap(self):
        West = ndimage.shift(self.seg,[-1,0,0],order=0)
        East = ndimage.shift(self.seg,[1,0,0],order=0)
        North = ndimage.shift(self.seg,[0,1,0],order=0)
        South = ndimage.shift(self.seg,[0,-1,0],order=0)
        Top = ndimage.shift(self.seg,[0,0,1],order=0)
        Bottom = ndimage.shift(self.seg,[0,0,-1],order=0)
        Sum = West + East + North + South + Top + Bottom
        ones = np.ones_like(self.seg)
        zeros = np.zeros_like(self.seg)
        Candidates = np.where(np.less(Sum,6),ones,zeros)
        Border = np.where(np.equal(np.multiply(Candidates,self.seg),1),ones,zeros)
        print(np.count_nonzero(Border))
        return Border

    def LabelBinary(self):
        blobs_labels = measure.label(self.seg, background=0)
        return blobs_labels
