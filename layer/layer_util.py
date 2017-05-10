def infer_spatial_rank(input_tensor):
    dims = input_tensor.get_shape().ndims - 2
    assert(dims > 0)
    return dims

#class RequireKeywords(object):
#    def __init__(self, *list_of_keys):
#        self.keys = list_of_keys
#
#    def __call__(self, f):
#        def wrapped(*args, **kwargs):
#            for key in self.keys:
#                if key not in kwargs:
#                    raise ValueError("{}: specify keywords: '{}'".format(
#                        args[0].layer_scope().name, self.keys))
#            return f(*args, **kwargs)
#        return wrapped
