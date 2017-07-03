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