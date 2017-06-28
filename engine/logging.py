# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
from tensorflow.core.framework import summary_pb2

CONSOLE='NiftyNetCollectionConsole'
LOG=tf.GraphKeys.SUMMARIES



def add_to_collections(keys,value):
  for k in keys:
    tf.add_to_collection(k,value)
	
def console_summary_string(byte_string):
  e=summary_pb2.Summary()
  e.ParseFromString(byte_string)
  try:
    return ', {}={:.8f}'.format(e.value[0].tag,e.value[0].simple_value)
  except:
    return ''
