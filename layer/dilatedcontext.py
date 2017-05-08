import tensorflow as tf
import layer_util
class DilatedTensor():
    """
    This context manager makes a wrapper of input_tensor
    When created, the input_tensor is dilated,
    the input_tensor resumes to original space when exiting the context.
    """
    def __init__(self, input_tensor, factor):
        self.tensor = input_tensor
        self.factor = factor
        self.spatial_rank = layer_util.infer_spatial_rank(self.tensor)
        self.zero_paddings = [[0, 0]] * self.spatial_rank
        self.block_shape=[factor] * self.spatial_rank
        for i in range(self.spatial_rank):
            assert(self.tensor.get_shape()[i+1] % factor == 0)

    def __enter__(self):
        if self.factor > 1:
            self.tensor = tf.space_to_batch_nd(
                    self.tensor, self.block_shape, self.zero_paddings)
        return self

    def __exit__(self, *args):
        if self.factor > 1:
            self.tensor = tf.batch_to_space_nd(
                    self.tensor, self.block_shape, self.zero_paddings)
