# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import six
import numpy as np
import tensorflow as tf

from niftynet.evaluation.segmentation_evaluator import SegmentationEvaluator
from niftynet.io.misc_io import set_logger
from tests.test_util import ParserNamespace

class SegmentationEvaluatorTest(tf.test.TestCase):
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
        self.assertEqual(result_dict[('subject_id', 'label')],
                      [{'subject_id':'test','label':1,'dice':1.}])
        self.assertEqual(result_dict[('subject_id', 'cc_id')],
                      [{'subject_id': 'test','cc_id': 'r1_s1',
                        'dice': 1.},
                       {'subject_id': 'test', 'cc_id': 'r2_s2',
                        'dice': 1.},
                       ])



if __name__ == "__main__":
    set_logger()
    # _run_test_application()
    tf.test.main()
