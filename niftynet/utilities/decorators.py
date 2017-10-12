from functools import wraps


def singleton(cls):
    """Decorate a class as singleton.

    Inspired by: https://wiki.python.org/moin/PythonDecoratorLibrary#Singleton
    """

    cls.__new_original__ = cls.__new__

    @wraps(cls.__new__)
    def singleton_new(cls_to_wrap, *args, **kw):
        it = cls_to_wrap.__dict__.get('__it__')
        if it is not None:
            return it

        cls_to_wrap.__it__ = it = cls_to_wrap.__new_original__(cls, *args, **kw)
        it.__init_original__(*args, **kw)
        return it

    cls.__new__ = singleton_new
    cls.__init_original__ = cls.__init__
    cls.__init__ = object.__init__

    return cls
