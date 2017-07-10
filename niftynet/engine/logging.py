# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
import warnings
from tensorflow.core.framework import summary_pb2

CONSOLE='NiftyNetCollectionConsole'
LOG=tf.GraphKeys.SUMMARIES



def add_to_collections(keys,value):
  for k in keys:
    tf.add_to_collection(k,value)

def console_summary_string(byte_string):
  try:
    e=summary_pb2.Summary()
    e.ParseFromString(byte_string)
    return ', {}={:.8f}'.format(e.value[0].tag,e.value[0].simple_value)
  except:
    warnings.warn('Summary could not be converted to string so it will not print on the command line')
    return ''

import numpy as np
import PIL
from PIL.GifImagePlugin import Image as GIF

def image3_animatedGIF(tag,ims):
  #x=numpy.random.randint(0,256,[10,10,10],numpy.uint8)
  ims = [np.asarray((ims[i,:,:]).astype(np.uint8)) for i in range(ims.shape[0])]
  ims=[GIF.fromarray(im) for im in ims]
  s=b''
  for b in PIL.GifImagePlugin.getheader(ims[0])[0]:
    s+=b
  s+=b'\x21\xFF\x0B\x4E\x45\x54\x53\x43\x41\x50\x45\x32\x2E\x30\x03\x01\x00\x00\x00'
  for i in ims:
    for b in PIL.GifImagePlugin.getdata(i):
      s+=b
  s+=b'\x3B'
  return [summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag,image=summary_pb2.Summary.Image(height=10,width=10,colorspace=1,encoded_image_string=s))]).SerializeToString()]

def image3(name,tensor,max_outputs=3,collections=[tf.GraphKeys.SUMMARIES],animation_axes=[1],image_axes=[2,3],other_indices={}):
  ''' Summary for higher dimensional images
  Parameters:
  name: string name for the summary
  tensor:   tensor to summarize. Should be in the range 0..255.
            By default, assumes tensor is NDHWC, and animates (through D)
            HxW slices of the 1st channel.
  collections: list of strings collections to add the summary to
  animation_axes=[1],image_axes=[2,3]
  '''
  if max_outputs==1:
    suffix='/image'
  else:
    suffix='/image/{}'
  axis_order = [0]+animation_axes+image_axes
  # slice tensor
  slicing = tuple((slice(None) if i in axis_order else slice(other_indices.get(i,0),other_indices.get(i,0)+1) for i in range(len(tensor.shape))))
  tensor=tensor[slicing]
  axis_order_all = axis_order+[i for i in range(len(tensor.shape.as_list())) if i not in axis_order]
  new_shape=[tensor.shape.as_list()[0],-1,tensor.shape.as_list()[axis_order[-2]],tensor.shape.as_list()[axis_order[-1]]]
  transposed_tensor = tf.reshape(tf.transpose(tensor,axis_order_all),new_shape)
  # split images
  with tf.device('/cpu:0'):
    for it in range(min(max_outputs,transposed_tensor.shape.as_list()[0])):
      T = tf.py_func(image3_animatedGIF,[name+suffix.format(it),transposed_tensor[it,:,:,:]],tf.string)
      [tf.add_to_collection(c,T) for c in collections]
  return T
def image3_sagittal(name,tensor,max_outputs=3,collections=[tf.GraphKeys.SUMMARIES]):
  return image3(name,tensor,max_outputs,collections,[1],[2,3])
def image3_coronal(name,tensor,max_outputs=3,collections=[tf.GraphKeys.SUMMARIES]):
  return image3(name,tensor,max_outputs,collections,[2],[1,3])
def image3_axial(name,tensor,max_outputs=3,collections=[tf.GraphKeys.SUMMARIES]):
  return image3(name,tensor,max_outputs,collections,[3],[1,2])
  