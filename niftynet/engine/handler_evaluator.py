# -*- coding: utf-8 -*-
"""
This module implements a sampler threads controller.
"""
import tensorflow as tf
import os
import nibabel as nib
import numpy as np

# from niftynet.engine.signal import SESS_STARTED
from niftynet.engine.signal import ITER_STARTED, ITER_FINISHED, SESS_STARTED, \
    SESS_FINISHED
from niftynet.utilities.util_common import traverse_nested
from niftynet.evaluation.pairwise_measures import PairwiseMeasures


class EvaluationHandler(object):
    """
    This class handles iteration events to start/stop samplers' threads.
    """

    def __init__(self, **_unused):
        # SESS_STARTED.connect(self.start_sampler_threads)
        self.internal_validation_flag = None
        SESS_STARTED.connect(self.start_evaluation_threads)
        SESS_FINISHED.connect(self.stop_evaluation_threads)
        ITER_FINISHED.connect(self.check_end_of_val_and_evaluate)

    def start_evaluation_threads(self, _sender, **_msg):
        pass

    def stop_evaluation_threads(self, _sender, **_msg):
        pass

    def check_end_of_val_and_evaluate(self, _sender, **_msg):
        iter_msg = _msg['iter_msg']

        if _sender.is_whole_volume_validating and iter_msg.is_validation:
            self.internal_validation_flag = True
        else:
            if self.internal_validation_flag:
                # In the last iteration we were in validation. Now we should evaluate
                # Search model dir.
                save_seg_dir = _sender.action_param.save_seg_dir
                for i in range(0, len(_sender.readers[-1].output_list)):
                    label = _sender.readers[-1](idx=i)[1]['label']
                    output = nib.load(
                        os.path.join(save_seg_dir,
                                     _sender.readers[-1]
                                     .get_subject_id(i) + '_wvv_out.nii.gz')) \
                        .get_data()
                    dice = \
                        PairwiseMeasures(
                            seg_img=np.where(output > 0, 1, 0),
                            ref_img=np.where(label > 0, 1, 0)).dice_score()
                    tf.logging.info(
                        'subject_id: {} dice: {}'.format(
                            _sender.readers[-1].get_subject_id(i), dice))
                self.internal_validation_flag = False
                # Match seg to ground truth
                # Compute PairwiseMeasures
                # Save results to TensorBoard/CSV
                pass
