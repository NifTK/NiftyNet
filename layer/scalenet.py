# -*- coding: utf-8 -*-
from six.moves import range

import tensorflow as tf
from .base import Layer
from .convolution import ConvolutionalLayer
from .highres3dnet import HighRes3DNet, HighResBlock


class ScaleNet(Layer):
    def __init__(self,
                 num_classes,
                 acti_type='prelu',
                 name='ScaleNet'):

        super(ScaleNet, self).__init__(name=name)
        self.n_features = 16
        self.num_classes = num_classes
        self.acti_type = acti_type
        self.name = "ScaleNet"
        print 'using {}'.format(self.name)


    def layer_op(self, images, is_training, layer_id=-1):
        n_modality = images.get_shape().as_list()[-1]
        rank = images.get_shape().ndims
        assert n_modality > 1
        roots = tf.split(images, n_modality, axis=rank-1)
        for (idx, root) in enumerate(roots):
            conv_layer = ConvolutionalLayer(n_output_chns=self.n_features,
                                            kernel_size=3,
                                            name='conv_{}'.format(idx))
            roots[idx] = conv_layer(root, is_training)
        roots = tf.stack(roots, axis=-1)

        back_end = ScaleBlock('AVERAGE', n_layers=1)
        output_tensor = back_end(roots, is_training)

        front_end = HighRes3DNet(self.num_classes)
        output_tensor = front_end(output_tensor, is_training)
        return output_tensor

SUPPORTED_OP = set(['MAX', 'AVERAGE'])
class ScaleBlock(Layer):
    def __init__(self, func, n_layers=1, name='scaleblock'):
        self.func = func.upper()
        assert self.func in SUPPORTED_OP
        super(ScaleBlock, self).__init__(name=name)
        self.n_layers = n_layers

    def layer_op(self, input_tensor, is_training):
        n_modality = input_tensor.get_shape().as_list()[-1]
        n_chns = input_tensor.get_shape().as_list()[-2]
        rank = input_tensor.get_shape().ndims
        perm = [i for i in range(rank)]
        perm[-2], perm[-1] = perm[-1], perm[-2]

        output_tensor = input_tensor
        for layer in range(self.n_layers):
            # modalities => feature channels
            output_tensor = tf.transpose(output_tensor, perm=perm)
            output_tensor = tf.unstack(output_tensor, axis=-1)
            for (idx, tensor) in enumerate(output_tensor):
                block_name = 'M_F_{}_{}'.format(layer, idx)
                highresblock_op = HighResBlock(n_output_chns=n_modality,
                                               kernels=(3, 1),
                                               with_res=True,
                                               name=block_name)
                output_tensor[idx] = highresblock_op(tensor, is_training)
                print highresblock_op
            output_tensor = tf.stack(output_tensor, axis=-1)

            # feature channels => modalities
            output_tensor = tf.transpose(output_tensor, perm=perm)
            output_tensor = tf.unstack(output_tensor, axis=-1)
            for (idx, tensor) in enumerate(output_tensor):
                block_name = 'F_M_{}_{}'.format(layer, idx)
                highresblock_op = HighResBlock(n_output_chns=n_chns,
                                               kernels=(3, 1),
                                               with_res=True,
                                               name=block_name)
                output_tensor[idx] = highresblock_op(tensor, is_training)
                print highresblock_op
            output_tensor = tf.stack(output_tensor, axis=-1)

        if self.func == 'MAX':
            output_tensor = tf.reduce_max(output_tensor, axis=-1)
        elif self.func == 'AVERAGE':
            output_tensor = tf.reduce_mean(output_tensor, axis=-1)
        return output_tensor
