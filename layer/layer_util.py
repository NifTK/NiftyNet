import numpy as np

def infer_spatial_rank(input_tensor):
    """
    e.g. given an input tensor [Batch, X, Y, Z, Feature] the spatial rank is 3
    """
    dims = input_tensor.get_shape().ndims - 2
    assert(dims > 0)
    return dims

def trivial_kernel(kernel_shape):
    """
    This function generates a trivial kernel with all 0s except for the
    element in its spatial center
    e.g. trivial_kernel((3, 3, 1, 1,)) returns a kernel of
    [[[[0]], [[0]], [[0]]],
     [[[0]], [[1]], [[0]]],
     [[[0]], [[0]], [[0]]]]
    kernel_shape[-1] and kernel_shape[-2] should be 1, so that it operates
    on the spatial dims only.  Note that there is no exact spatial centre
    if np.any((kernel_shape % 2) != 1), however it is fine in many cases
    as np.sum(trivial_kernel(kernel_shape)) == 1
    """
    assert(kernel_shape[-1] == 1)
    assert(kernel_shape[-2] == 1)
    #assert(np.all((kernel_shape % 2) == 1))
    kernel = np.zeros(kernel_shape)
    flattened = kernel.reshape(-1)
    flattened[np.prod(kernel_shape)//2] = 1
    return flattened.reshape(kernel_shape)

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
