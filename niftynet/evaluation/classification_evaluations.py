# -*- coding: utf-8 -*-
"""
This module defines built-in evaluation functions for classification 
applications

Many classification metrics only make sense computed over all subjects,
so aggregation is used.
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

from niftynet.evaluation.base_evaluations import BaseEvaluation
from niftynet.evaluation.base_evaluator import ScalarAggregator,\
                                               DataFrameAggregator

class accuracy(BaseEvaluation):
    def layer_op(self, subject_id, data):
        metric_name = 'accuracy_'
        if self.app_param.output_prob:
            inferred_label = np.argmax(data['inferred'][0,0,0,0,:])
        else:
            inferred_label = data['inferred'][0,0,0,0,0]
        pdf = pd.DataFrame.from_records([{'subject_id':subject_id,
                                          'acc_i':inferred_label,
                                          'acc_l':data['label'][0,0,0,0,0]}],
                                        index=('subject_id',))
        return [pdf]
        
    def aggregate(self, df):
        print(df)
        agg = pd.DataFrame.from_records([{'accuracy':(df.acc_i==df.acc_l).mean()}])
        return [agg]

    def get_aggregations(self):
        return [DataFrameAggregator(('subject_id',), self.aggregate)]
