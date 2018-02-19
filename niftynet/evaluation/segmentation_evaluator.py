# -*- coding: utf-8 -*-
"""
This module defines the specialized Evaluator for segmentation applications
All logic except default metrics is delegated to the parent class
"""

from __future__ import absolute_import, division, print_function

from niftynet.evaluation.base_evaluator import CachedSubanalysisEvaluator

class SegmentationEvaluator(CachedSubanalysisEvaluator):
    """
    Evaluator for SegmentationApplication
    Supports caching of intermediate results which is
    important for boundary error calculations
    """
    def default_evaluation_list(self):
        """
        :return:  list of metric names to compute by default
        """
        return ['dice', 'jaccard', 'average_distance']
