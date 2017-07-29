# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from niftynet.io.input_volume import MultimodalVolume
import pandas
import os
import tensorflow as tf

class VolumeReader(object):
    def __init__(self):
        self.file_list = None

    def initialise_reader(self, data_param, task_param):
        app_type = task_param.name
        self.file_list = self.load_file_list(data_param)

        if app_type == "net_segmentation.py":
            import pdb; pdb.set_trace()
            pass

    def load_file_list(self, data_param):
        file_list = None
        for modality_name in data_param:
            csv_file = data_param.get(modality_name, '').csv_file
            if not os.path.isfile(csv_file):
                tf.logging.fatal("csv file {} not found.".format(csv_file))
                raise IOError
            csv_list = pandas.read_csv(
                csv_file, header=None, names=['id', modality_name])
            if file_list is None:
                file_list = csv_list
                continue
            # merge file_list based on ids (first column of each csv)
            n_rows = file_list.shape[0]
            file_list = pandas.merge(file_list, csv_list, on='id')
            if file_list.shape[0] != n_rows:
                tf.logging.warning("rows not matched in {}".format(csv_file))
        return file_list
