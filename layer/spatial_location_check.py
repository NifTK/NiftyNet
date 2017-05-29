import tensorflow as tf
import numpy as np

from .base import Layer


class SpatialLocationCheckLayer(Layer):
    """
    validate spatial location against provided discrete segmentation map
    and constraints on the map
    """

    def __init__(self,
                 discrete_volume,
                 compulsory=[[0], [0]],
                 minimum_ratio=0.01,
                 min_numb_labels=1,
                 padding=0,
                 name='spatial_location_check'):
        super(SpatialLocationCheck, self).__init__(name=name)
        self.discrete_volume = discrete_volume
        self.compulsory_labels = compulsory[0]
        self.compulsory_ratios = compulsory[1]
        self.min_ratio = max(minimum_ratio, 0)
        self.min_nlabels = min_numb_labels
        self.padding = max(padding, 0)

        self._volume_worth_checking = self.__volume_level_eval()

    def __volume_level_eval(self):
        uni = np.unique(np.array(self.discrete_volume).flatten())
        if len(uni) < self.min_nlabels:
            return False
        for x in self.compulsory_labels:
            if x not in uni:
                return False


    def layer_op(self, location, spatial_rank):
        if not self._volume_worth_checking:
            return location

        if spatial_rank == 3:
            xs, ys, zs = location[0:spatial_rank] + self.padding
            xe, ye, ze = location[spatial_rank:] - self.padding
            test_cube = self.discrete_volume[xs:xe, ys:ye, zs:ze, ...]
        elif spatial_rank == 2:
            xs, ys = location[0:spatial_rank] + self.padding
            xe, ye = location[spatial_rank:] - self.padding
            test_cube = self.discrete_volume[xs:xe, ys:ye, ...]

        uniq_, count_ = np.unique(np.asarray(test_cube).flatten(),
                                  return_counts=True)
        n_total = float(test_cube.size)
        ratio_ = count_ / n_total
        if len(self.compulsory_labels) > 0:
            for (x, r) in zip(self.compulsory_labels, self.compulsory_values):
                if x not in uniq_:
                    return False
                if ratio_[uniq_.index(x)] < r:
                    return False

        is_enough = ratio_ > self.min_ratio
        is_compulsory = [x in self.compulsory for x in uniq_]
        satisfied = np.asarray(is_enough, dtype=bool) +\
                np.asarray(is_compulsory, dtype=bool)
        return np.sum(satisfied) > min_nlabels:
