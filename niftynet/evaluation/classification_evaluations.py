# -*- coding: utf-8 -*-
"""
This module defines built-in evaluation functions for classification 
applications

Many classification metrics only make sense computed over all subjects,
so aggregation is used.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from niftynet.evaluation.base_evaluations import BaseEvaluation, \
    ResultsDictionary
from niftynet.evaluation.base_evaluator import ScalarAggregator

class accuracy(BaseEvaluation):
    def layer_op(self, subject_id, data):
        metric_name = 'accuracy_'
        if self.app_param.output_prob:
            inferred_label = np.amax(data['inferred'][0,0,0,0,:])
            metric_value = (inferred_label, data['label'][0,0,0,0,0])
        else:
            metric_value = (data['inferred'][0,0,0,0,0],
                            data['label'][0,0,0,0,0])
        results_dict = ResultsDictionary()
        results_dict[('subject_id',)] = [{'subject_id':subject_id,
                                          metric_name:metric_value}]
        return results_dict

    def get_aggregations(self):
        def agg_func(values):
          #print(values)
          return sum([1 if v[0]==v[1] else 0 for v in values])/len(values)
        aggregations = []
        agg = ScalarAggregator('accuracy_',
                               ('subject_id',),
                               (), agg_func,
                               'accuracy')
        return [agg]