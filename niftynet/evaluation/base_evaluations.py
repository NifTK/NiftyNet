# -*- coding: utf-8 -*-
"""
This module defines basic interfaces for NiftyNet evaluations
"""

from __future__ import absolute_import, division, print_function

from collections import defaultdict

import pandas as pd

from niftynet.layer.base_layer import Layer

class ResultsDictionary(defaultdict):
    """
    The class represents a set of calculated metrics.
    It is a defaultdict of pandas DataFrames

    The dictionary is indexed by the dataframe objects' indices, which
    should uniquely define the row of the result table for the metrics
    contained within.

    For example:
    {('subject_id','label'):[{'subject_id':'foo','label':2,'score':2,'val':3}]}

    These constraints are not programmatically enforced.
    """
    def __init__(self, data=None):
        super(ResultsDictionary, self).__init__(lambda: pd.DataFrame())
        if data is None:
            return
    
        if isinstance(data, pd.DataFrame):
            key = tuple(data.index.names)
            self[key]=self[key].combine_first(data)
        else:
            for datum in data:
                if isinstance(datum, pd.DataFrame):
                    key = tuple(datum.index.names)
                    self[key]=self[key].combine_first(datum)
    def update_all(self, new_results):
        for group_by in new_results:
            self[group_by]=self[group_by].combine_first(new_results[group_by])

class BaseEvaluation(Layer):
    """
    Minimal interface for a NiftyNet evaluation
    """
    def __init__(self, reader, app_param, eval_param):
        super(BaseEvaluation, self).__init__()
        self.reader = reader
        self.app_param = app_param
        self.eval_param = eval_param

    def get_aggregations(self):
        """
        Returns aggregations to compute for the metric. Each aggregation is
        a callable that modifies and returns a ResultsDictionary object. See
        BaseEvaluator.ScalarAggregator for an example.

        :return: list of aggregation callables
        """
        return []

    def layer_op(self, subject_id, data):
        """
        Perform one evaluation calculation for one subject
        :param subject_id:  subject identifier string
        :param data:  a data dictionary as built by ImageReader
        :return: a ResultsDictionary object
        """
        raise NotImplementedError('not implemented in abstract base class')

class CachedSubanalysisEvaluation(BaseEvaluation):
    """
    Interface for NiftyNet evaluations used with
    CachedSubanalysisEvaluator so that evaluations are run in a way that is
    friendly for caching intermediate computations. Each evaluation defines
    sub-analyses to run, and all subanalysis are run at the same time then
    the cache is cleared
    """
    def subanalyses(self, subject_id, data):
        """
        This function defines the sub-analyses to run. All evaluations with
        matching sub-analyses will be run in sequence, before clearing the cache
        :param subject_id:  subject identifier string
        :param data:  a data dictionary as built by ImageReader
        :return: list of dictionaries, each containing information specifyng
        the analysis to run. Elements will be passed to layer_op one at a
        time in a cache friendly order
        """
        raise NotImplementedError('not implemented in abstract class')

    def layer_op(self, subject_id, data, subanalysis):
        """
        Performs one sub-analysis

        :param subject_id: subject identifier string
        :param data: a data dictionary as built by ImageReader
        :param subanalysis: dictionary containing information specifying the
        analysis to run
        :return: a ResultsDictionary object
        """
        raise NotImplementedError('not implemented in abstract class')
