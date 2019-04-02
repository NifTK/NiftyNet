import tensorflow as tf
from niftynet.engine.handler_gradient import ApplyGradients, _apply_gradients


class GuunetApplyGradients(ApplyGradients):

    def __init__(self, is_training_action=False, **_):
        super().__init__()


    def add_gradients(self, sender, **msg):
        if msg['iter_msg'].is_training:
            if msg['iter_msg'].


