# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

class Loadable(ABCMeta):
    """
    interface of loadable data
    """
    @abstractmethod
    def load_as_5D_matrix(self):
        raise NotImplementedError


class DataFromFile(Loadable):
    def __init__(self):
        self._data = None
        self._name = None
        self._file_path = None


class Volume(DataFromFile):
    def __init__(self):
        super(Volume, self).__init__()
        self._interp_order = None
        self._orientation = None
        self._affine = None
        self._pixdim = None


class MultimodalVolume(Loadable):
    def __init__(self):
        pass
