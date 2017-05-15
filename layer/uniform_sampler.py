# -*- coding: utf-8 -*-
import numpy as np
from six.moves import range

from .base_sampler import BaseSampler


class UniformSampler(BaseSampler):
    """
    This class generators samples by uniformly sampling each input volume
    """

    def __init__(self, patch, name="uniform_sampler"):
        super(UniformSampler, self).__init__(patch=patch, name=name)

    def layer_op(self, batch_size=1):
        # batch_size is needed here so that it generates total number of
        # N samples where (N % batch_size) == 0
