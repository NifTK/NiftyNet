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
        if _iter_msg.is_training and not _iter_msg.is_validation:
            return
        _sender.is_whole_volume_validating = True
        # _sender.action = INFER

    def run_whole_volume_validation(self, _sender, **msg):
        _iter_msg = msg['iter_msg']
        if _iter_msg.is_training:
            return
        # Must make sure current_iter hasn't been incremented
        _iter_msg = msg['iter_msg']