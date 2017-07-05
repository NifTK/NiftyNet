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

def image3_animatedGIF(tag,x):
  #x=numpy.random.randint(0,256,[10,10,10],numpy.uint8)
  ims=[GIF.fromarray(np.asarray((x[0,i,:,:,0]).astype(np.uint8))) for i in range(x.shape[3])]
  s=b''
  for b in PIL.GifImagePlugin.getheader(ims[0])[0]:
    s+=b
  s+=b'\x21\xFF\x0B\x4E\x45\x54\x53\x43\x41\x50\x45\x32\x2E\x30\x03\x01\x00\x00\x00'
  for i in ims:
    for b in PIL.GifImagePlugin.getdata(i):
      s+=b
  s+=b'\x3B'
  return [summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=tag,image=summary_pb2.Summary.Image(height=10,width=10,colorspace=1,encoded_image_string=s))]).SerializeToString()]

def image3(tag,x,collections=[tf.GraphKeys.SUMMARIES]):
  with tf.device('/cpu:0'):
    T = tf.py_func(image3_animatedGIF,[tag,x],tf.string)
  [tf.add_to_collection(c,T) for c in collections]
  return T
