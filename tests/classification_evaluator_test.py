# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import six
import numpy as np
import tensorflow as tf

from niftynet.evaluation.classification_evaluator import ClassificationEvaluator
from niftynet.io.misc_io import set_logger
from tests.test_util import ParserNamespace

class ClassificationEvaluatorTest(tf.test.TestCase):
    def test_basic(self):
        class NS(object):
            def __init__(self, dict):
                self.__dict__.update(dict)
        classification_param=NS({'num_classes':2,
                                 'output_prob':False})
        eval_param=NS({'evaluations':'niftynet.evaluation.classification_evaluations.accuracy'})
        positive = np.reshape(1,[1,1,1,1,1])
        negative = np.reshape(0,[1,1,1,1,1])
        mask = np.reshape(np.abs(np.linspace(0.,2.,64)-1)>.8,[4,4,4,1,1])
        tp = {'label':positive,'inferred':positive}
        fp = {'label':negative,'inferred':positive}
        tn = {'label':negative,'inferred':negative}
        fn = {'label':positive,'inferred':negative}
        interp_orders = {'label':0,'inferred':-1}
        image_folder = '.'
        e = ClassificationEvaluator(None, classification_param, eval_param)

        def generator():
            yield ('test', tp,interp_orders)
            yield ('test', tp,interp_orders)
            yield ('test', fn,interp_orders)
            yield ('test', fp,interp_orders)

        result_dict = e.evaluate_from_generator(generator())
        self.assertIn((), result_dict)
        self.assertEqual(result_dict[()],
                      [{'accuracy': 0.5}])



if __name__ == "__main__":
    set_logger()
    # _run_test_application()
    tf.test.main()
