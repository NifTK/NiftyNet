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

from niftynet.evaluation.base_evaluations import BaseEvaluation,\
    ResultsDictionary
from niftynet.evaluation.base_evaluator import ScalarAggregator,\
                                               DataFrameAggregator

class accuracy(BaseEvaluation):
    def layer_op(self, subject_id, data):
        metric_name = 'accuracy_'
        if self.app_param.output_prob:
            inferred_label = np.amax(data['inferred'][0,0,0,0,:])
        else:
            inferred_label = data['inferred'][0,0,0,0,0]
        pdf = pd.DataFrame.from_records([{'subject_id':subject_id,
                                          'acc_i':inferred_label,
                                          'acc_l':data['label'][0,0,0,0,0]}],
                                        index=('subject_id',))
        return ResultsDictionary(pdf)
        
    def aggregate(self, df):
        agg = pd.DataFrame.from_records([{'accuracy':(df.acc_i==df.acc_l).mean()}])
        print(agg)
        return agg

    def get_aggregations(self):
        return [DataFrameAggregator(('subject_id',), self.aggregate)]

class roc(BaseEvaluation):
    def layer_op(self, subject_id, data):
        if not self.app_param.output_prob or\
           self.app_param.num_classes>2:
           return ResultsDictionary()
        pdf = pd.DataFrame.from_records([{'subject_id':subject_id,
                            'roc_i':data['inferred'][0,0,0,0,1],
                            'roc_l':data['label'][0,0,0,0,0]}],('subject_id',))
        return ResultsDictionary(pdf)

    @classmethod
    def aggregate(cls, df):
        thresholds = np.linspace(0,1,10)
        df_out = pd.DataFrame(index=range(len(thresholds)),columns=('thresholds','tp','fp','tn','fn','acc','sens','spec'))
        df_out.thresholds = thresholds
        for it in range(len(thresholds)):
            df_out.loc[it,'tp'] = (df[df.roc_l==1].roc_i>it).sum()
            df_out.loc[it,'fp'] = (df[df.roc_l==0].roc_i>it).sum()
            df_out.loc[it,'tn'] = (df[df.roc_l==0].roc_i<=it).sum()
            df_out.loc[it,'fn'] = (df[df.roc_l==1].roc_i<=it).sum()
        df_out.acc = (df_out.tp+df_out.tn)/(
            df_out.tp+df_out.tn+df_out.fp+df_out.fn)
        denom = df_out.tp+df_out.fn
        df_out.loc[denom>0,'sens'] = df_out.loc[denom>0,'tp']/denom[denom>0]
        denom = df_out.tn+df_out.fp
        df_out.loc[denom>0,'spec'] = df_out.loc[denom>0,'tn']/denom[denom>0]
        df_out=df_out.set_index('thresholds')
        return df_out

    def get_aggregations(self):
        return [DataFrameAggregator(('subject_id',), self.aggregate)]

class roc_auc(BaseEvaluation):
    def layer_op(self, subject_id, data):
        if not self.app_param.output_prob or\
           self.app_param.num_classes>2:
           return ResultsDictionary()
        pdf = pd.DataFrame.from_records([{'subject_id':subject_id,
                            'roc_auc_i':data['inferred'][0,0,0,0,1],
                            'roc_auc_l':data['label'][0,0,0,0,0]}],('subject_id',))
        return ResultsDictionary(pdf)


    def aggregate(self, df):
        by_threshold = roc.aggregate(df)
        tpr = np.array(list(by_threshold.sens))
        tnr = 1 - np.array(by_threshold.spec)
        
        roc_auc = np.sum((tpr[:-1] + tpr[1:]) * (tnr[1:] - tnr[:-1]) / 2)
        return pd.DataFrame.from_records([{'roc_auc':roc_auc}])

    def get_aggregations(self):
        return [DataFrameAggregator(('subject_id',), self.aggregate)]
