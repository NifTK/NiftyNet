import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data.util import nest

from niftynet.layer.base_layer import Layer

class CSVReader(Layer):
    
    def __init__(self, names=None):
        self._paths = None
        self._labels = None
        self._df = None
        self.label_names = None
        self.dims = None
        
        super(CSVReader, self).__init__(name='csv_reader')
    
    def initialise(self, path_to_csv):
        label_df = pd.read_csv(path_to_csv, header=None, names=['subject_ids', 'labels'])
        self._paths = label_df['subject_ids'].values
        self.label_names = list(label_df['labels'].unique())
        self._df = label_df
        self.dims = len(self.label_names)
        
        self._labels = self.to_ohe(label_df['labels'].values)
        return self
    
    def to_ohe(self, labels):
        return [np.eye(len(self.label_names))[self.label_names.index(label)] for label in labels]
    
    def layer_op(self, idx=None, shuffle=True):
        # def apply_expand_dims(x, n):
        #     if n==0:
        #         return x
        #     return np.expand_dims(apply_expand_dims(x, n - 1), -1)
        data = self._labels[idx]
        while len(data.shape) < 4:
            data = np.expand_dims(data, -1)
        label_dict = {'label': data}
        # label_dict = {'label': apply_expand_dims(np.expand_dims(np.array(data).astype(np.float32), 0), 4)}
        return idx, label_dict, None
    
    @property
    def shapes(self):
        """
        :return: dict of label shape and label location shape
        """
        self._shapes = {'label': (1, self.dims, 1, 1, 1, 1), 'label_location': (1, 7)}
        return self._shapes
    
    @property
    def tf_dtypes(self):
        """
        Infer input data dtypes in TF
        """
        self._dtypes = {'label': tf.float32, 'label_location': tf.int32}
        return self._dtypes
    
    @property
    def tf_shapes(self):
        """
        :return: a dictionary of sampler output tensor shapes
        """
        output_shapes = nest.map_structure_up_to(
            self.tf_dtypes, tf.TensorShape, self.shapes)
        return output_shapes