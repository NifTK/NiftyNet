import tensorflow as tf
import numpy as np
from six.moves import range

from base_net import BaseNet
from scalenet import ScaleNet
from highres3dnet import HighResBlock
from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.convolution import ConvolutionalLayer
from niftynet.layer.convolution import ConvLayer
from niftynet.layer.bn import BNLayer
from niftynet.layer.loss import LossFunction
from niftynet.layer.dilatedcontext import DilatedTensor
from niftynet.layer.downsample import DownSampleLayer
from niftynet.layer.upsample import UpSampleLayer

# Distance matrix between labels of BraTS dataset defined manually
# they are used to calculate the Wasserstein Dice loss
M_tree = np.array([ [0., 1., 1., 1., 1.],
                    [1., 0., 0.6, 0.2, 0.5],
                    [1., 0.6, 0., 0.6, 0.7],
                    [1., 0.2, 0.6, 0., 0.5],
                    [1., 0.5, 0.7, 0.5, 0.] ], dtype=np.float32)

M_01 = np.array([[0., 1., 1., 1., 1.],
                 [1., 0., 1., 1., 1.],
                 [1., 1., 0., 1., 1.],
                 [1., 1., 1., 0., 1.],
                 [1., 1., 1., 1., 0.]], dtype=np.float32)

# not used for BrainLes paper
M_13 = np.array([[0., 1., 1., 1., 1.],
                 [1., 0., 1., 0., 1.],
                 [1., 1., 0., 1., 1.],
                 [1., 0., 1., 0., 1.],
                 [1., 1., 1., 1., 0.]], dtype=np.float32)

class HolisticScaleNet(ScaleNet):
    def __init__(self,
                 num_classes,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 acti_func='elu',
                 name='HolisticScaleNet'
                 ):
        super(HolisticScaleNet, self).__init__(
                                               num_classes=num_classes,
                                               acti_func=acti_func,
                                               name=name,
                                               w_initializer=w_initializer,
                                               w_regularizer=w_regularizer,
                                               b_initializer=b_initializer,
                                               b_regularizer=b_regularizer,
                                               )
        
        self.num_scale_res_block = 0
        self.num_res_blocks = [3, 3, 3, 3]
        self.num_features = [70]*4
        self.num_fea_score_layers = [[70, 140]]*4

        # self.loss = LossFunction(num_classes, loss_type='Dice', decay=0.0)

    def layer_op(self, input_tensor, is_training, layer_id=-1):
        # BaseNet._print_activations(images)
        zero_paddings = [[0, 0], [0, 0], [0, 0]]
        num_scales = 5
        is_training = True
        layer_instances = []
        scores_instances = []
        first_conv_layer = ConvolutionalLayer(n_output_chns=self.num_features[0],
                                      with_bn=True,
                                      kernel_size=3,
                                      w_initializer=self.initializers['w'],
                                      w_regularizer=self.regularizers['w'],
                                      acti_func=self.acti_func,
                                      name='conv_1_1')
        flow = first_conv_layer(input_tensor, is_training)
        layer_instances.append((first_conv_layer, flow))

        with DilatedTensor(flow, dilation_factor=1) as dilated:
            for j in range(self.num_res_blocks[0]):
                res_block = HighResBlock(
                    self.num_features[0],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % ('res_1', j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        # SCALE 1
        score_layer_scale1 = ScoreLayer(
            num_features=self.num_fea_score_layers[0],
            num_classes=self.num_classes)
        score_1 = score_layer_scale1(flow, is_training)
        scores_instances.append(score_1)
        # if is_training:
        #     loss_s1 = WGDL(score_1, labels)
        #     tf.add_to_collection('multiscale_loss', loss_s1/num_scales)



        # # SCALE 2
        with DilatedTensor(flow, dilation_factor=2) as dilated:
            for j in range(self.num_res_blocks[1]):
                res_block = HighResBlock(
                    self.num_features[1],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % ('res_2', j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        score_layer_scale2 = ScoreLayer(
            num_features=self.num_fea_score_layers[1],
            num_classes=self.num_classes)
        score_2 = score_layer_scale2(flow,is_training)


        # score_2 = self.score_layer(flow, self.num_fea_score_layers[1])
        up_score_2 = score_2
        scores_instances.append(up_score_2)
        # if is_training:
        #     loss_s2 =  self.WGDL(score_2, labels)
        #     # loss_s2 = self.new_dice_loss(score_2, labels)
        #     tf.add_to_collection('multiscale_loss', loss_s2/num_scales)


        # SCALE 3
        ## dowsampling factor = 2
        downsample_scale3 = DownSampleLayer(func='AVG',kernel_size=2,\
                                                                      stride=2)
        flow = downsample_scale3(flow)
        layer_instances.append((downsample_scale3,flow))
        with DilatedTensor(flow, dilation_factor=1) as dilated:
            for j in range(self.num_res_blocks[2]):
                res_block = HighResBlock(
                    self.num_features[2],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % ('res_3', j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        score_layer_scale3 = ScoreLayer(
            num_features=self.num_fea_score_layers[2],
            num_classes=self.num_classes)
        score_3 = score_layer_scale3(flow, is_training)

        upsample_indep_scale3 = UpSampleLayer(func='CHANNELWISE_DECONV',
                                       kernel_size=2,
                                       stride=2,
                                       w_initializer=tf.constant_initializer(1.0, dtype=tf.float32))
        up_score_3 = upsample_indep_scale3(score_3)
        scores_instances.append(up_score_3)

        # up_score_3 = self.feature_indep_upsample_conv(score_3, factor=2)
        # if is_training:
        #     loss_s3 = self.WGDL(up_score_3, labels)
        #     # loss_s3 = self.new_dice_loss(up_score_3, labels)
        #     tf.add_to_collection('multiscale_loss', loss_s3/num_scales)


        # SCALE 4
        with DilatedTensor(flow, dilation_factor=2) as dilated:
            for j in range(self.num_res_blocks[3]):
                res_block = HighResBlock(
                    self.num_features[3],
                    acti_func=self.acti_func,
                    w_initializer=self.initializers['w'],
                    w_regularizer=self.regularizers['w'],
                    name='%s_%d' % ('res_4', j))
                dilated.tensor = res_block(dilated.tensor, is_training)
                layer_instances.append((res_block, dilated.tensor))
        flow = dilated.tensor

        score_layer_scale4 = ScoreLayer(
            num_features=self.num_fea_score_layers[3],
            num_classes=self.num_classes)
        score_4 = score_layer_scale4(flow, self.num_fea_score_layers[3],is_training)

        upsample_indep_scale4 = UpSampleLayer(func='CHANNELWISE_DECONV',
                                              kernel_size=1,
                                              stride=2,
                                              w_initializer=tf.constant_initializer(
                                                  1.0, dtype=tf.float32))
        up_score_4 = upsample_indep_scale4(score_4)
        scores_instances.append(up_score_4)

        # if is_training:
        #     loss_s4 = self.WGDL(up_score_4, labels)
        #     # loss_s4 = self.new_dice_loss(up_score_4, labels)
        #     tf.add_to_collection('multiscale_loss', loss_s4/num_scales)

        # FUSED SCALES
        merge_layer = MergeLayer('WEIGHTED_AVERAGE')
        soft_scores = []
        for s in scores_instances:
            soft_scores.append(tf.nn.softmax(s))
        fused_score = merge_layer(soft_scores)
        scores_instances.append(fused_score)
        # with tf.variable_scope('fusion'):
        #     # softmax is apply before merging to normalise the different predictions
        #     # and allow to compute a weighted sum probabilistic segmentations
        #
        #     fused_score = self._merge_roots([tf.nn.softmax(score_1), tf.nn.softmax(up_score_2),
        #                                      tf.nn.softmax(up_score_3), tf.nn.softmax(up_score_4)],
        #                                      merging_type='weighted_average')
            # self._print_activations(fused_score)
            # if is_training:
            #     fused_loss = self.WGDL(fused_score, labels)
            #     # fused_loss = self.new_dice_loss(fused_score, labels)
            #     tf.add_to_collection('multiscale_loss', fused_loss/num_scales)
        if is_training:
            return scores_instances
        else:
            return flow


class ScoreLayer(TrainableLayer):
    def __init__(self,
                 num_features=None,
                 w_initializer=None,
                 w_regularizer=None,
                 num_classes=1,
                 acti_func='elu',
                 name='ScoreLayer'):
        super(ScoreLayer, self).__init__(name=name)
        self.num_classes = num_classes
        self.acti_func = acti_func
        self.num_features = num_features
        self.n_layers = len(self.num_features)
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self, input_tensor, is_training, layer_id=-1):
        n_modality = input_tensor.get_shape().as_list()[-1]
        n_chns = input_tensor.get_shape().as_list()[-2]
        rank = input_tensor.get_shape().ndims
        perm = [i for i in range(rank)]
        perm[-2], perm[-1] = perm[-1], perm[-2]
        output_tensor = input_tensor
        n_layers = self.n_layers
        # All layers except the last one consists in:
        # BN + Conv_3x3x3 + Activation
        # layer_instances = []

        for layer in range(n_layers - 1):
            layer_to_add = ConvolutionalLayer(
                n_output_chns=self.num_features[layer+1],
                with_bn=True,
                kernel_size=3,
                w_initializer=self.initializers['w'],
                w_regularizer=self.regularizers['w'],
                acti_func=self.acti_func,
                name='conv_fc_%d' % layer)
            output_tensor = layer_to_add(output_tensor, is_training)
            # layer_instances.append((layer_to_add, output_tensor))
        last_layer = ConvolutionalLayer(n_output_chns=self.num_classes,
                                        kernel_size=1)
        output_tensor = last_layer(output_tensor, is_training)
        # layer_instances.append((last_layer, output_tensor))
        return output_tensor



    # def __init_variable(self, name, shape, init, trainable=True, withreg=True):
    #     with tf.device('/%s:0' % self._device_string):
    #         var = tf.get_variable(  # init variable if not exists
    #             name, shape, initializer=init, trainable=trainable)
    #         if trainable and withreg:
    #             tf.add_to_collection('reg_var', var)
    #     return var
    #
    # def __variable_with_weight_decay(self, name, shape, stddev):
    #     # !!!check with default settings
    #     # TODO this if-else tree needs to be redesigned...
    #     if name == 'const':
    #         return self.__init_variable(
    #             name, shape,
    #             tf.constant_initializer(0.0, dtype=tf.float32),
    #             trainable=True, withreg=False)
    #     elif name == 'b': # default bias initialised to 0
    #         return self.__init_variable(
    #             name, shape,
    #             tf.constant_initializer(0.0, dtype=tf.float32),
    #             trainable=True, withreg=True)
    #     elif (name == 'w') and (stddev < 0): #default weights initialiser
    #         stddev = np.sqrt(1.3 * 2.0 / (np.prod(shape[:-2])*shape[-1]))
    #         return self.__init_variable(
    #             name, shape,
    #             tf.truncated_normal_initializer(
    #                 mean=0.0, stddev=stddev, dtype=tf.float32),
    #             trainable=True, withreg=True)
    #     elif name == 'w':  # initialiser with custom stddevs
    #         return self.__init_variable(
    #             name, shape,
    #             tf.truncated_normal_initializer(
    #                 mean=0.0, stddev=stddev, dtype=tf.float32),
    #             trainable=True, withreg=True)
    #     return None

SUPPORTED_OPS = {'AVERAGE','WEIGHTED_AVERAGE','MAXOUT'}

class MergeLayer(TrainableLayer):
    def __init__(self,
                 func,
                 w_initializer=None,
                 w_regularizer=None,
                 acti_func='elu',
                 name='MergeLayer'):
        super(MergeLayer, self).__init__(name=name)
        self.func = func
        # self.num_classes = num_classes
        self.acti_func = acti_func
        # self.num_features = num_features
        # self.n_layers = len(self.num_features)
        self.initializers = {'w': w_initializer}
        self.regularizers = {'w': w_regularizer}

    def layer_op(self,roots):
        if self.func == 'MAXOUT':
            return tf.reduce_max(tf.stack(roots,axis=-1),axis=-1)
        elif self.func == 'AVERAGE':
            return tf.reduce_mean(tf.stack(roots,axis=-1),axis=-1)
        elif self.func == 'WEIGHTED_AVERAGE':
            input_tensor = tf.stack(roots,axis=-1)
            n_modality = input_tensor.get_shape().as_list()[-1]
            n_chns = input_tensor.get_shape().as_list()[-2]
            rank = input_tensor.get_shape().ndims
            perm = [i for i in range(rank)]
            perm[-2], perm[-1] = perm[-1], perm[-2]

            output_tensor = input_tensor
            output_tensor = tf.transpose(output_tensor, perm=perm)
            output_tensor = tf.unstack(output_tensor, axis=-1)
            n_roots = len(roots)
            roots_merged = []
            # fea_roots = tf.unstack(tf.stack(roots,axis=-1),axis=-2)
            n_fea = len(output_tensor)
            for f in range(n_fea):
                conv_layer = ConvLayer(n_output_chns=1,kernel_size=1,
                                       stride=1)
                roots_merged_f = conv_layer(output_tensor[f])
                roots_merged.append(roots_merged_f)
            return tf.concat(roots_merged,axis=-1)
            #
            # for layer in range(self.n_layers):
            #     # modalities => feature channels
            #     output_tensor = tf.transpose(output_tensor, perm=perm)
            #     output_tensor = tf.unstack(output_tensor, axis=-1)
            #     for (idx, tensor) in enumerate(output_tensor):
            #         block_name = 'M_F_{}_{}'.format(layer, idx)
            #         highresblock_op = HighResBlock(
            #             n_output_chns=n_modality,
            #             kernels=(3, 1),
            #             with_res=True,
            #             w_initializer=self.initializers['w'],
            #             w_regularizer=self.regularizers['w'],
            #             acti_func=self.acti_func,
            #             name=block_name)
            #         output_tensor[idx] = highresblock_op(tensor, is_training)
            #         print(highresblock_op)
            #     output_tensor = tf.stack(output_tensor, axis=-1)
            #
            #     # feature channels => modalities
            #     output_tensor = tf.transpose(output_tensor, perm=perm)
            #     output_tensor = tf.unstack(output_tensor, axis=-1)
            #     for (idx, tensor) in enumerate(output_tensor):
            #         block_name = 'F_M_{}_{}'.format(layer, idx)
            #         highresblock_op = HighResBlock(
            #             n_output_chns=n_chns,
            #             kernels=(3, 1),
            #             with_res=True,
            #             w_initializer=self.initializers['w'],
            #             w_regularizer=self.regularizers['w'],
            #             acti_func=self.acti_func,
            #             name=block_name)
            #         output_tensor[idx] = highresblock_op(tensor, is_training)
            #         print(highresblock_op)
            #     output_tensor = tf.stack(output_tensor, axis=-1)
            #
            #
            #
            # n_roots = len(roots)
            # roots_merged = []
            # fea_roots = tf.unstack(tf.stack(roots,axis=-1),axis=-2)
            # n_fea = len(fea_roots)
            # for f in range(n_fea):
            #     conv_layer = ConvLayer(kernel_size=1,stride=1)
            #     roots_merged_f = conv_layer(fea_roots[f])
            #     roots_merged.append(roots_merged_f)
            # return tf.concat(roots_merged,axis=-1)


6

    # def feature_indep_upsample_conv(self, f_in, factor=2):
    #     # trainable enlarging of the spatial dims by the given factor
    #     # for each feature independently and without mixing them (learned)
    #     i_dim = [i.value for i in f_in.get_shape()]
    #     unstack_f_in = tf.unstack(f_in, axis=4)
    #     unstack_f_up = []
    #     c = -1
    #     for fc_in in unstack_f_in:
    #         c += 1
    #         with tf.variable_scope('upsample_c%d' % c):
    #             fc_in =tf.expand_dims(fc_in, axis=4)
    #             kernel = self.__init_variable(
    #                         'w', shape=[factor,factor,factor,1,1],
    #                         init=tf.constant_initializer(1.0, dtype=tf.float32),
    #                         trainable=True, withreg=False)
    #             fc_in = tf.nn.conv3d_transpose(
    #                         fc_in, kernel,
    #                         [i_dim[0], i_dim[1]*factor, i_dim[2]*factor, i_dim[3]*factor, 1],
    #                         [1, factor, factor, factor, 1], padding='SAME')
    #             unstack_f_up.append(fc_in)
    #     f_up = tf.concat(unstack_f_up, axis=4)
    #     return f_up
    #
    # # not currently used
    # def unpool(self, f_in, factor, ni_=None):
    #     # enlarge the spatial dims by a given factor by linear interpolation (not learned)
    #     i_dim = [i.value for i in f_in.get_shape()]
    #     if ni_ is None:
    #         ni_ = i_dim[-1]
    #     kernel_np = np.zeros(shape=[factor,factor,factor,ni_,ni_])
    #     upsample_kernel = np.ones(shape=[factor,factor,factor])
    #     for i in range(ni_):
    #         # don't mix different chanels
    #         kernel_np[:,:,:,i,i] = upsample_kernel
    #     kernel = tf.constant(kernel_np, dtype=tf.float32)
    #     up_conv = tf.nn.conv3d_transpose(
    #         f_in, kernel,
    #         [i_dim[0], i_dim[1]*factor, i_dim[2]*factor, i_dim[3]*factor, ni_],
    #         [1, factor,factor,factor, 1], padding='SAME')
    #     return up_conv
