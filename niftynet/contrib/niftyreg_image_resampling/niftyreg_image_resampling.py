from __future__ import print_function, division

import tensorflow as tf
from tensorflow.python.framework import ops
from niftynet.layer.base_layer import Layer
from niftynet.layer.layer_util import infer_spatial_rank

from niftyreg_module_loader import get_niftyreg_module


# NiftyNet boundary types to NiftyReg code mapping
__BOUNDARY_CODES__ = {
    'ZERO': 0,
    'NAN': 1,
    'REPLICATE': 2,
    'SYMMETRIC': 3
}


# Exposure of supported boundary types for compat. w/ ResamplerLayer
SUPPORTED_BOUNDARY = {k for k in __BOUNDARY_CODES__}


# NiftyNet interpolation types to NiftyReg code mapping
__INTERPOLATION_CODES__ = {'NEAREST': 0,
                           'LINEAR': 1,
                           'BSPLINE': 3}


# Exposure of supported interpolation types for compat. w/ ResamplerLayer
SUPPORTED_INTERPOLATION = {k for k in __INTERPOLATION_CODES__}


# NiftyReg expects displacement components to be
# indexed w/ slowest index
def _transpose(data):
    nof_dims = len(data.shape) - 1
    perm = [0] + list(range(nof_dims, 0, -1))
    perm += list(range(nof_dims + 1, len(data.shape)))
    assert len(perm) == len(data.shape)

    return tf.transpose(data, perm)


@ops.RegisterGradient("NiftyregImageResampling")
def _niftyreg_resampling_grad(op, grad):
    grad_op = get_niftyreg_module().niftyreg_image_resampling_gradient(
        op.inputs[0],
        op.inputs[1],
        interpolation=op.get_attr('interpolation'),
        boundary=op.get_attr('boundary'))

    chained_grad = None

    nof_modalities = op.inputs[0].shape.as_list()[1]
    if not nof_modalities is None and nof_modalities != 1:
        nof_dims = op.inputs[1].shape.as_list()[1]

        assert grad_op.shape.as_list()[1] == nof_modalities*nof_dims

        chained_grads = []
        for m in range(nof_modalities):
            mod_grad = grad_op[:,(nof_dims*m):((m+1)*nof_dims),...]
            out_mod_grad = tf.expand_dims(grad[:,m,...], axis=1)

            out_grad = tf.tile(out_mod_grad, [1] + [nof_dims]
                               + [1]*(len(grad_op.shape) - 2))

            chained_grads.append(tf.multiply(mod_grad, out_grad))

        chained_grad = tf.reduce_sum(tf.stack(chained_grads, axis=0), axis=0)

    else:
        grad_rep = tf.tile(grad, [1] + [grad_op.shape[1]]
                           + [1]*(len(grad_op.shape) - 2))
        chained_grad = tf.multiply(grad_rep, grad_op)

    image_grad_op \
        = get_niftyreg_module().niftyreg_image_resampling_image_gradient(
            op.inputs[0],
            op.inputs[1],
            grad,
            interpolation=op.get_attr('interpolation'),
            boundary=op.get_attr('boundary'))

    return [image_grad_op, chained_grad]


class NiftyregImageResamplingLayer(Layer):
    def __init__(self, interpolation, boundary='ZERO', **kwargs):
        super(NiftyregImageResamplingLayer, self).__init__(**kwargs)

        self._interpolation = __INTERPOLATION_CODES__[interpolation.upper()]
        self._boundary = boundary.upper()

    def layer_op(self, inputs, deformation, **kwargs):
        nof_dims = infer_spatial_rank(inputs)
        nof_output_dims = infer_spatial_rank(deformation)

        batch_size = inputs.shape.as_list()[0]
        if deformation.shape.as_list()[0] != batch_size:
            deformation = tf.tile(deformation,
                                  [batch_size] + [1]*(nof_output_dims + 1))

        output_spatial_dims = deformation.shape.as_list()[1:-1]
        input_dims = [d if d else -1 for d in inputs.shape.as_list()]
        if len(output_spatial_dims) != nof_dims:
            resample_def = deformation
            while len(resample_def.shape) < len(inputs.shape):
                resample_def = tf.expand_dims(resample_def,
                                              axis=len(resample_def.shape) - 2)
        else:
            resample_def = deformation
        assert infer_spatial_rank(resample_def) == nof_dims

        resampled = get_niftyreg_module().niftyreg_image_resampling(
            _transpose(inputs),
            _transpose(resample_def),
            interpolation=self._interpolation,
            boundary=__BOUNDARY_CODES__[self._boundary])

        return tf.reshape(
            _transpose(resampled),
            [batch_size] + output_spatial_dims + [input_dims[-1]])

