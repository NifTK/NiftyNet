import os

import tensorflow as tf
from niftynet.engine.signal import \
TRAIN, INFER, ITER_STARTED, ITER_FINISHED, GRAPH_CREATED


class WholeVolumeValidation(object):
    """
        This class does not handle whole volume validation
    """

    def __init__(self,
                 **_unused):
        self.current_iter = None
        GRAPH_CREATED.connect(self.setup_logging_directories)
        ITER_STARTED.connect(self.change_to_inference)
        ITER_FINISHED.connect(self.run_whole_volume_validation)

    def setup_logging_directories(self, _sender, **_unused_msg):
        pass

    def change_to_inference(self, _sender, **msg):
        _iter_msg = msg['iter_msg']
        self.current_iter = _iter_msg.current_iter
        if _iter_msg.is_training and not _iter_msg.is_validation:
            return
        # _sender.is_whole_volume_validating = True
        _sender.action = INFER

    def run_whole_volume_validation(self, _sender, **msg):
        _iter_msg = msg['iter_msg']
        if _iter_msg.is_training:
            return
        self.revert_to_training(_sender, **msg)

    def revert_to_training(self, _sender, **msg):
        # Must make sure current_iter hasn't been incremented
        _iter_msg = msg['iter_msg']
        _iter_msg.current_iter = self.current_iter
        # Set application flag back to is_training
        # _sender.is_whole_volume_validating = False
        if not _iter_msg.is_validation:
            # _sender.is_training = True
            _sender.action = TRAIN
