# -*- coding: utf-8 -*-
"""
This module implements a sampler threads controller.
"""
import tensorflow as tf

# from niftynet.engine.signal import SESS_STARTED
from niftynet.engine.signal import ITER_STARTED, ITER_FINISHED, SESS_STARTED, SESS_FINISHED
from niftynet.utilities.util_common import traverse_nested


class EvaluationHandler(object):
    """
    This class handles iteration events to start/stop samplers' threads.
    """

    def __init__(self, **_unused):
        # SESS_STARTED.connect(self.start_sampler_threads)
        self.internal_validation_flag = None
        SESS_STARTED.connect(self.start_evaluation_threads)
        SESS_FINISHED.connect(self.stop_evalutation_threads)
        ITER_FINISHED.connect(self.check_end_of_val_and_evaluate)

    def start_evaluation_threads(self):
        pass

    def stop_evaluation_threads(self):
        pass

    def check_end_of_val_and_evaluate(self, _sender, **_msg):
        iter_msg = _msg['iter_msg']

        if _sender.is_whole_volume_validating and iter_msg.is_validation:
            self.internal_validation_flag = True
        else:
            if self.internal_validation_flag:
                # In the last iteration we were in validation. Now we should evaluate
                # Search model dir.
                # Match seg to ground truth
                # Compute PairwiseMeasures
                # Save results to TensorBoard/CSV
                pass


