# -*- coding: utf-8 -*-
"""
This module defines built-in evaluation functions for regression applications

"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

from niftynet.evaluation.base_evaluations import BaseEvaluation


class BaseRegressionEvaluation(BaseEvaluation):
    """ Interface for scalar regression metrics """
    def layer_op(self, subject_id, data):
        metric_name = self.__class__.__name__
        metric_value = self.metric(data['inferred'], data['output'])
        pdf = pd.DataFrame.from_records([{'subject_id':subject_id,
                                          metric_name:metric_value}],
                                        ('subject_id',))
        return [pdf]

    def metric(self, reg, ref):
        """
        Computes a scalar value for the metric
        :param reg: np.array with inferred regression
        :param ref: np array with the reference output
        :return: scalar metric value
        """
        raise NotImplementedError

#pylint: disable=invalid-name
class mse(BaseRegressionEvaluation):
    """ Computes mean squared error """
    def metric(self, reg, ref):
        return np.mean(np.square(reg - ref))


class rmse(BaseRegressionEvaluation):
    """ Computes root mean squared error """
    def metric(self, reg, ref):
        return  np.sqrt(np.mean(np.square(reg - ref)))


class mae(BaseRegressionEvaluation):
    """ Computes mean absolute error """
    def metric(self, reg, ref):
        return np.mean(np.abs(ref - reg))
