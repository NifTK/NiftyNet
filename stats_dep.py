import scipy
import nibabel
import numpy as np
import numpy.ma as ma
import tensorflow as tf
import scipy.stats.mstats as mstats
from util import MorphologyOps

class perform_statistics(object):
  def __init__(self, seg_img, data_img, num_neighbors, threshold=0, pixdim=[1,1,1]):
      self.seg = seg_img
      self.data = data_img
      self.neigh = num_neighbors
      self.pixdim = pixdim
      self.threshold = threshold

  def CoM(self):
      SegBin = np.copy(self.seg)
      SegBin[SegBin>self.threshold] = 1
      SegBin[SegBin<=self.threshold] = 0
      list_coordinates = np.argwhere(SegBin == 1)
      return np.mean(list_coordinates,0)

  def Vol(self):
      vol_vox = np.prod(self.pixdim)
      numb_seg = np.sum(self.seg)
      numb_seg_bin = np.count_nonzero(self.seg)
      return numb_seg, numb_seg_bin,\
             numb_seg*vol_vox, numb_seg_bin*vol_vox

  def Surface(self):
      vol_vox = np.prod(self.pixdim)
      border_seg = MorphologyOps(self.seg,self.neigh).border_map()
      numb_border_seg_bin = np.count_nonzero(border_seg)
      numb_border_seg = np.sum(border_seg)
      return numb_border_seg, numb_border_seg_bin, \
             numb_border_seg * vol_vox, numb_border_seg_bin * vol_vox

  def SAV(self):
      Sn, Snb,Sv,Svb = self.Surface()
      Vn, Vnb,Vv,Vvb = self.Vol()
      return Sn/Vn, Snb/Vnb, Sv/Vv, Svb/Vvb

  def Compactness(self):
      Sn, Snb,Sv,Svb = self.Surface()
      Vn, Vnb,Vv,Vvb = self.Vol()
      return np.power(Sn,1.5)/Vn,np.power(Snb,1.5)/Vnb,np.power(Sv,1.5)/Vv,np.power(Svb,1.5)/Vvb,

  def Min(self):
      mask_temp = 1-self.seg
      mask_temp[mask_temp<0.5] = 0
      mask_temp[mask_temp>=0.5] = 1
      mask = np.tile(mask_temp,[1,1,1,self.data.shape[3]])
      masked_array = ma.masked_array(self.data,mask)
      new_arr = masked_array.reshape(-1, masked_array.shape[-1])
      return ma.min(new_arr,0)

  def Max(self):
      mask_temp = 1-self.seg
      mask_temp[mask_temp<0.5] = 0
      mask_temp[mask_temp>=0.5] = 1
      mask = np.tile(mask_temp,[1,1,1,self.data.shape[3]])
      masked_array = ma.masked_array(self.data,mask)
      new_arr = masked_array.reshape(-1, masked_array.shape[-1])
      return ma.max(new_arr,0)

  def Average(self):
      mask_temp = 1-self.seg
      mask_temp[mask_temp<0.5] = 0
      mask_temp[mask_temp>=0.5] = 1
      mask = np.tile(mask_temp,[1,1,1,self.data.shape[3]])
      masked_array = ma.masked_array(self.data,mask)
      new_arr = masked_array.reshape(-1, masked_array.shape[-1])
      weights = np.tile(self.seg, [1, 1, 1, self.data.shape[3]]).reshape(-1, masked_array.shape[-1])
      return ma.average(new_arr,axis=0,weights=weights)

  def Mean(self):
      mask_temp = 1-self.seg
      mask_temp[mask_temp<0.5] = 0
      mask_temp[mask_temp>=0.5] = 1
      mask = np.tile(mask_temp,[1,1,1,self.data.shape[3]])
      masked_array = ma.masked_array(self.data,mask)
      new_arr = masked_array.reshape(-1, masked_array.shape[-1])
      return ma.mean(new_arr,0)

  def Median(self):
      mask_temp = 1-self.seg
      mask_temp[mask_temp<0.5] = 0
      mask_temp[mask_temp>=0.5] = 1
      mask = np.tile(mask_temp,[1,1,1,self.data.shape[3]])
      masked_array = ma.masked_array(self.data,mask)
      new_arr = masked_array.reshape(-1, masked_array.shape[-1])
      return ma.median(new_arr,0 )

  def Skewness(self):
      mask_temp = 1-self.seg
      mask_temp[mask_temp<0.5] = 0
      mask_temp[mask_temp>=0.5] = 1
      mask = np.tile(mask_temp,[1,1,1,self.data.shape[3]])
      masked_array = ma.masked_array(self.data,mask)
      new_arr = masked_array.reshape(-1, masked_array.shape[-1])
      return mstats.skew(new_arr,0)

  def SD(self):
      mask_temp = 1-self.seg
      mask_temp[mask_temp<0.5] = 0
      mask_temp[mask_temp>=0.5] = 1
      mask = np.tile(mask_temp,[1,1,1,self.data.shape[3]])
      masked_array = ma.masked_array(self.data,mask)
      new_arr = masked_array.reshape(-1, masked_array.shape[-1])
      return ma.std(new_arr,0)

  def Kurtosis(self):
      mask_temp = 1-self.seg
      mask_temp[mask_temp<0.5] = 0
      mask_temp[mask_temp>=0.5] = 1
      mask = np.tile(mask_temp,[1,1,1,self.data.shape[3]])
      masked_array = ma.masked_array(self.data,mask)
      new_arr = masked_array.reshape(-1, masked_array.shape[-1])
      return mstats.kurtosis(new_arr,0)

  def Quantiles(self):
      mask_temp = 1 - self.seg
      mask_temp[mask_temp < 0.5] = 0
      mask_temp[mask_temp >= 0.5] = 1
      mask = np.tile(mask_temp, [1, 1, 1, self.data.shape[3]])
      masked_array = ma.masked_array(self.data, mask)
      new_arr = masked_array.reshape(-1, masked_array.shape[-1])
      return mstats.mquantiles(new_arr, axis=0)
