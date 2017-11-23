# -*- coding: utf-8 -*-


class ParserNamespace(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
