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
from morphology import get_morphology

class perform_evaluation(object):
  def __init__(self, seg_img, ref_img,data_img, num_neighbors, pixdim=[1,1,1]):
      self.seg = seg_img
      self.ref = ref_img
      self.data = data_img
      self.neigh = num_neighbors
      self.pixdim = pixdim

  def FPmap(self):
      init = self.seg-self.ref
      zeros = np.zeros_like(self.seg)
      ones = np.ones_like(self.seg)
      FPmap = np.where(np.greater(init,0),ones,zeros)
      return FPmap

  def FNmap(self):
      init = self.ref - self.seg
      zeros = np.zeros_like(self.seg)
      ones = np.ones_like(self.seg)
      FNmap = np.where(np.greater(init, 0), ones, zeros)
      return FNmap

  def TPmap(self):
      init = self.ref + self.seg
      zeros = np.zeros_like(self.seg)
      ones = np.ones_like(self.seg)
      TPmap = np.where(np.greater(init, 1), ones, zeros)
      return TPmap

  def TNmap(self):
      init = -self.ref - self.seg
      zeros = np.zeros_like(self.seg)
      ones = np.ones_like(self.seg)
      TNmap = np.where(np.greater(init,-0.5),ones,zeros)
      return TNmap

  def Unionmap(self):
      init = self.ref + self.seg
      zeros = np.zeros_like(self.seg)
      ones = np.ones_like(self.seg)
      Unionmap = np.where(np.greater(init,0.5),ones,zeros)
      return Unionmap

  def ConnectedElements(self):
      init = np.multiply(self.seg, self.ref)
      GMRef = get_morphology(self.ref, self.neigh)
      GMSeg = get_morphology(self.seg, self.neigh)
      blobs_ref = GMRef.LabelBinary()
      blobs_seg = GMSeg.LabelBinary()
      list_blobs_ref = np.unique(blobs_ref[blobs_ref > 0])
      list_blobs_seg = np.unique(blobs_seg[blobs_seg > 0])
      mul_blobs_ref = np.multiply(blobs_ref, init)
      mul_blobs_seg = np.multiply(blobs_seg, init)
      list_TP_ref = np.unique(mul_blobs_ref[mul_blobs_ref > 0])
      list_TP_seg = np.unique(mul_blobs_seg[mul_blobs_seg > 0])
      list_FN = [x for x in list_blobs_ref if x not in list_TP_ref]
      list_FP = [x for x in list_blobs_seg if x not in list_TP_seg]
      return len(list_TP_ref), len(list_FP), len(list_FN)

  def ConnectedErrorMaps(self):
      init = np.multiply(self.seg, self.ref)
      GMRef = get_morphology(self.ref,self.neigh)
      GMSeg = get_morphology(self.seg, self.neigh)
      blobs_ref = GMRef.LabelBinary()
      blobs_seg = GMSeg.LabelBinary()
      list_blobs_ref = np.unique(blobs_ref[blobs_ref>0])
      list_blobs_seg = np.unique(blobs_seg[blobs_seg>0])
      mul_blobs_ref = np.multiply(blobs_ref, init)
      mul_blobs_seg = np.multiply(blobs_seg, init)
      list_TP_ref = np.unique(mul_blobs_ref[mul_blobs_ref>0])
      list_TP_seg = np.unique(mul_blobs_seg[mul_blobs_seg>0])
      list_FN = [x for x in list_blobs_ref if x not in list_TP_ref]
      list_FP = [x for x in list_blobs_seg if x not in list_TP_seg]
      ones = np.ones_like(blobs_ref)
      TPcMap = np.zeros_like(blobs_ref)
      FPcMap = np.zeros_like(blobs_ref)
      FNcMap = np.zeros_like(blobs_ref)
      print(np.max(blobs_ref),np.max(blobs_seg))
      for i in list_TP_ref:
          #print(i)
          TPcMap[blobs_ref==i]=1
          #np.where(np.equal(blobs_ref,i),ones,TPcMap)
      for i in list_TP_seg:
          TPcMap[blobs_seg==i]=1
      print(np.count_nonzero(TPcMap))
      for i in list_FN:
          FNcMap[blobs_ref == i] = 1
      for i in list_FP:
          FPcMap[blobs_seg == i] = 1
      return TPcMap,FNcMap,FPcMap

  def OE(self):
      TPcMap, _, _ = self.ConnectedErrorMaps()
      OEFMap = self.ref - np.multiply(TPcMap,self.seg)
      unique, counts = np.unique(OEFMap, return_counts=True)
      print(counts)
      OEFN = counts[unique==1]
      OEFP = counts[unique==-1]
      if len(OEFN)==0:
          OEFN = 0
      if len(OEFP)==0:
          OEFP = 0
      OER = 2*(OEFN+OEFP)/(np.sum(self.seg)+np.sum(self.ref))
      return OER, OEFP, OEFN

  def DE(self):
      TPcMap, FNcMap, FPcMap = self.ConnectedErrorMaps()
      DEFN = np.sum(FNcMap)
      DEFP = np.sum(FPcMap)
      return DEFN+DEFP, DEFP,DEFN

  def FNc(self):
      init = np.multiply(self.seg, self.ref)
      blobs = get_morphology.LabelBinary(self.ref)
      mul_blobs = np.multiply(blobs, init)
      list_blobs = np.unique(mul_blobs)
      return len(list_blocks)

  def TN(self):
      TNmap = self.TNmap()
      TN = np.sum(TNmap)
      return TN

  def TP(self):
      TPmap = self.TPmap()
      TP = np.sum(TPmap)
      return TP

  def FP(self):
      FPmap = self.FPmap()
      FP = np.sum(FPmap)
      return FP

  def FN(self):
      FNmap = self.FNmap()
      FN = np.sum(FNmap)
      return FN

  def Sensitivity(self):
      Sens = np.sum(self.TPmap())/np.sum(self.ref)
      return Sens

  def Specificity(self):
      Spec = np.sum(self.TNmap())/np.sum(1-self.ref)
      return Spec

  def Accuracy(self):
      Acc = (self.TN()+self.TP())/\
            (self.TN()+self.TP()+
             self.FN()+self.FP())
      return Acc

  def PPV(self):
      PPV = self.TP()/(self.TP()+self.FP())
      return PPV

  def NPV(self):
      NPV = self.TN()/(self.FN()+self.TN())
      return NPV

  def FPR(self):
      FPR = self.FP()/np.sum(1-self.ref)
      return FPR

  def DSC(self):
      DSC = 2*self.TP()/np.sum(self.ref+self.seg)
      return DSC

  def Jaccard(self):
      Jaccard = np.sum(np.multiply(self.ref,self.seg))/(np.sum(self.Unionmap()))
      return Jaccard

  def Informedness(self):
      return self.Sensitivity() + self.Specificity() -1

  def Markedness(self):
      return self.PPV() + self.NPV() -1

  def VolDiff(self):
      VolDiff = np.abs(np.sum(self.ref)-np.sum(self.seg))/np.sum(self.ref)
      return VolDiff

  def AvDist(self):
      dist = DistanceMetric.get_metric('euclidean')
      GMRef = get_morphology(self.ref,self.neigh)
      GMSeg = get_morphology(self.seg,self.neigh)
      border_ref = GMRef.BorderMap()
      border_seg = GMSeg.BorderMap()
      coord_ref = np.multiply(np.argwhere(border_ref>0),self.pixdim)
      coord_seg = np.multiply(np.argwhere(border_seg>0),self.pixdim)
      pairwise_dist = dist.pairwise(coord_ref,coord_seg)
      AvDist = (np.sum(np.min(pairwise_dist,0))+np.sum(np.min(pairwise_dist,1)))/(np.sum(self.ref+self.seg))
      return AvDist

  def HD(self):
      dist = DistanceMetric.get_metric('euclidean')
      GMRef = get_morphology(self.ref,self.neigh)
      GMSeg = get_morphology(self.seg,self.neigh)
      border_ref = GMRef.BorderMap()
      border_seg = GMSeg.BorderMap()
      coord_ref = np.multiply(np.argwhere(border_ref>0),self.pixdim)
      coord_seg = np.multiply(np.argwhere(border_seg>0),self.pixdim)
      pairwise_dist = dist.pairwise(coord_ref,coord_seg)
      HD = np.max([np.max(np.min(pairwise_dist,0)),np.max(np.min(pairwise_dist,1))])
      return HD





  