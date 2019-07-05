# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.evaluation.segmentation_evaluator import SegmentationEvaluator
from niftynet.io.misc_io import set_logger
from tests.niftynet_testcase import NiftyNetTestCase

class SegmentationEvaluatorTest(NiftyNetTestCase):
    def test_basic(self):
        class NS(object):
            def __init__(self, dict):
                self.__dict__.update(dict)
        segmentation_param=NS({'evaluation_units':'foreground,cc',
                               'num_classes':2,
                            'output_prob':False})
        eval_param=NS({'evaluations':'Dice'})
        mask = np.reshape(np.abs(np.linspace(0.,2.,64)-1)>.8,[4,4,4,1,1])
        data_dict = {'label':mask,'inferred':mask}
        interp_orders = {'label':0,'inferred':0}
        image_folder = '.'
        e = SegmentationEvaluator(None, segmentation_param, eval_param)
        
        def generator():
            yield ('test',data_dict,interp_orders)

        result_dict = e.evaluate_from_generator(generator())
        self.assertIn(('subject_id', 'label'), result_dict)
        self.assertIn(('subject_id', 'cc_id'), result_dict)
        self.assertEqual(tuple(result_dict[('subject_id', 'label')].index.names),
                         ('subject_id', 'label'))
        self.assertEqual(tuple(result_dict[('subject_id', 'cc_id')].index.names),
                         ('subject_id', 'cc_id'))
        print(result_dict[('subject_id', 'cc_id')].to_dict('index'))
        self.assertEqual(result_dict[('subject_id', 'label')].to_dict('index'),
                      {('test', 1): {'dice': 1.}})
        self.assertEqual(result_dict[('subject_id', 'cc_id')].to_dict('index'),
                      {('test', 'r1_s1'): {'dice': 1.},
                       ('test', 'r2_s2'): {'dice': 1.}})


if __name__ == "__main__":
    set_logger()
    # _run_test_application()
    tf.test.main()
