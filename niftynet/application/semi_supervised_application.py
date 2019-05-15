import tensorflow as tf


from niftynet.application.base_application import BaseApplication
from niftynet.engine.application_factory import ApplicationNetFactory
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import NETWORK_OUTPUT
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.engine.sampler_linear_interpolate_v2 import LinearInterpolateSampler
from niftynet.engine.sampler_resize_v2 import ResizeSampler
from niftynet.engine.windows_aggregator_identity import WindowAsImageAggregator
from niftynet.io.image_reader import ImageReader

from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer

#import niftynet.layer.loss_autoencoder as nnlla
from niftynet.layer.vae_loss import VAELossLayer
import niftynet.layer.loss_segmentation as nnlls
import niftynet.layer.loss_uncertainty as nnllu
import niftynet.layer.loss_kl_divergence as nnllk

from niftynet.utilities.util_common import look_up_operations
from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.engine.sampler_grid_v2 import GridSampler
from niftynet.engine.application_factory import InitializerFactory

import math

SUPPORTED_INPUT = set(['image', 'label'])
SUPPORTED_TRAINING = set(['label', 'no_label'])


class SemiSupervisedApplication(BaseApplication):
    REQUIRED_CONFIG_SECTION = "SEMISUPERVISED"

    #TODO: Needs enumeration of operations (segmentation/classification, etc.)
    def __init__(self, net_param, action_param, action):
        super(SemiSupervisedApplication, self).__init__()
        tf.logging.info('starting semi-supervised application')
        self.action = action

        self.net_param = net_param
        self.action_param = action_param

        self.data_param = None
        self.semi_supervised_param = None
        self.seg_weight = None
        self.kl_weight = None
        self.vae_weight = None

        self.has_seg_feature = True
        self.has_seg_logvar_decoder = False#True
        self.has_autoencoder_feature = True
        self.has_autoencoder_logvar_decoder = False#True

        self.learning_rate = None
        self.current_id = None
        self.base_lr = self.action_param.lr
        self.lr_decay_base = self.action_param.lr_decay_base
        self.lr_decay_rate = self.action_param.lr_decay_rate

    def initialise_dataset_loader(self,
                                  data_param=None,
                                  task_param=None,
                                  data_partitioner=None):

        self.data_param = data_param
        self.semi_supervised_param = task_param
        self.seg_weight = self.semi_supervised_param.seg_weight
        self.kl_weight = self.semi_supervised_param.kl_weight
        self.vae_weight = self.semi_supervised_param.vae_weight
        # self.has_seg_feature = self.semi_supervised_param.enable_seg
        # self.has_seg_logvar_decoder = True
        # self.has_autoencoder_feature = self.semi_supervised_param.enable_vae
        # self.has_autoencoder_logvar_decoder = True

        if self.is_training:
            reader_names = ('image', 'label')
        elif self.is_inference:
            reader_names = ('image',)
        elif self.is_evaluation:
            reader_names = ('image', 'label')
        else:
            tf.logging.fatal(
                'Action `%s` not supported. Expected one of %s',
                self.action, self.SUPPORTED_PHASES)
            raise ValueError

        try:
            reader_phase = self.action_param.dataset_to_infer
        except AttributeError:
            reader_phase = None

        file_lists = data_partitioner.get_file_lists_by(phase=reader_phase,
                                                        action=self.action)

        self.readers = [
            ImageReader(reader_names).initialise(data_param,
                                                 task_param,
                                                 flist, True) for flist in file_lists
        ]

        #self.readers[0].layer_op(idx=0)

        preprocessing = []

        if self.net_param.whitening:
            preprocessing.append(MeanVarNormalisationLayer(
                image_name='image',
                #binary_masking_func=foreground_masking_layer
            ))

        if self.net_param.histogram_ref_file and self.net_param.normalisation:
            preprocessing.append(HistogramNormalisationLayer(
                image_name='image',
                modalities=vars(task_param).get('image'),
                model_filename=self.net_param.histogram_ref_file,
                #binary_masking_func=foreground_masking_layer,
                norm_type=self.net_param.norm_type,
                cutoff=self.net_param.cutoff,
                name='hist_norm_layer'))
        if len(preprocessing) > 0:
            for r in self.readers:
                r.add_preprocessing_layers(preprocessing)


    def initialise_sampler(self):
        if self.is_training:
            self.sampler =\
                [[UniformSampler(
                    reader=reader,
                    window_sizes=self.data_param,
                    batch_size=self.net_param.batch_size,
                    windows_per_image=self.action_param.sample_per_volume,
                    queue_length=self.net_param.queue_length,
                    pad_if_smaller=True)
                        for reader in self.readers]]
        else:
            self.sampler =\
                [[GridSampler(
                    reader=reader,
                    window_sizes=self.data_param,
                    batch_size=self.net_param.batch_size,
                    spacial_window_size=self.action_param.spacial_window_size,
                    queue_length=self.net_param.queue_length,
                    pad_if_smaller=True)
                        for reader in self.readers]]


    def initialise_network(self):
        w_regularizer = None
        b_regularizer = None
        reg_type = self.net_param.reg_type.lower()
        decay = self.net_param.decay
        if reg_type == 'l2' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l2_regularizer(decay)
            b_regularizer = regularizers.l2_regularizer(decay)
        elif reg_type == 'l1' and decay > 0:
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l1_regularizer(decay)
            b_regularizer = regularizers.l1_regularizer(decay)

        self.net = ApplicationNetFactory.create(self.net_param.name)(
            num_classes=self.semi_supervised_param.num_classes,
            w_initializer=InitializerFactory.get_initializer(
                name=self.net_param.weight_initializer),
            b_initializer=InitializerFactory.get_initializer(
                name=self.net_param.bias_initializer),
            w_regularizer=w_regularizer,
            b_regularizer=b_regularizer,
            has_seg_feature=self.has_seg_feature,
            has_seg_logvar_decoder=self.has_seg_logvar_decoder,
            has_autoencoder_feature=self.has_autoencoder_feature,
            has_autoencoder_logvar_decoder=self.has_autoencoder_logvar_decoder,
            acti_func=self.net_param.activation_function)


    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):


        def switch_sampler(for_training):
            #TODO: NN: are there any situations where there is more than one sampler
            #layer?
            #TODO: NN: this is pure boilerplate and shouldn't be rewritten for
            #every application
            with tf.name_scope('train' if for_training else 'validation'):
                sampler = self.get_sampler()[0][0 if for_training else -1]
                #TODO: NN: pop_batch_op doesn't pop; this name is misleading
                return sampler.pop_batch_op()


        def rescale_image(image, source_min, source_max, dest_max=255):
            tclip = tf.clip_by_value(image, source_min, source_max)
            return (tclip - source_min) / (source_max - source_min) * dest_max



        if self.is_training:

            #self.sampler[0][0].layer_op(idx=0)
            #print("sampler =", self.sampler)

            if self.action_param.validation_every_n > 0:
                data_dict = tf.cond(tf.logical_not(self.is_validation),
                                    lambda: switch_sampler(for_training=True),
                                    lambda: switch_sampler(for_training=False))
            else:
                data_dict = switch_sampler(for_training=True)

            image = tf.cast(data_dict['image'], tf.float32)
            #image = rescale_image(image, 0.0, 100.0, 1.0)
            net_args = {
                'is_training': self.is_training
            }
            outputs, internal = self.net(image, **net_args)
            #segmentation_output = outputs[]


            with tf.name_scope('Optimiser'):
                self.learning_rate = tf.placeholder(tf.float32, shape=[])

                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.learning_rate)

            #loss function
            seg_loss = None
            if self.has_seg_feature:
                # seg_means = outputs['final_seg_output']
                # seg_logvars = tf.constant(0.005, dtype=tf.float32, shape=seg_means.shape)
                seg_means = outputs['seg_means']
                sm_seg_means = tf.nn.softmax(seg_means)
                seg_logvars =\
                    outputs.get('seg_logvars', None)
                seg_truth = data_dict.get('label', None)
                seg_loss_func = nnllu.LossFunction('l2', name='seg_uncertainty_loss')
                seg_losses = seg_loss_func(
                    prediction_means=tf.reshape(seg_means, [-1, seg_means.shape[-1]]),
                    prediction_logvars=None if seg_logvars is None else tf.reshape(seg_logvars, [-1, seg_logvars.shape[-1]]),
                    ground_truth=tf.reshape(seg_truth, [-1, 1]))
                    #ground_truth = tf.squeeze(tf.reshape(seg_truth, [-1, 1])))
                seg_loss = tf.reduce_mean(seg_losses)

            image_loss = None
            kl_loss = None
            autoencoder_loss = None
            if self.has_autoencoder_feature:
                # image_means = outputs['final_image_output']
                # image_logvars = tf.constant(0.005, dtype=tf.float32, shape=image_means.shape)
                image_means = outputs['image_means']
                image_logvars =\
                    outputs.get('image_logvars', None)
                image_truth = data_dict.get('image', None)
                image_loss_func = nnllu.LossFunction('l2', name='image_uncertainty_loss')
                image_losses = image_loss_func(
                    prediction_means=tf.reshape(image_means, [-1, image_means.shape[-1]]),
                    prediction_logvars=None if image_logvars is None else tf.reshape(image_logvars, [-1, image_logvars.shape[-1]]),
                    ground_truth=tf.reshape(image_truth, [-1, image_truth.shape[-1]]))
                image_loss = tf.reduce_mean(image_losses)

                kl_means = outputs['posterior_means']
                kl_logvars = outputs['posterior_logvars']
                kl_loss_func = nnllk.LossFunction(name='kl_loss')
                kl_losses = kl_loss_func(means=kl_means, logvars=kl_logvars)
                kl_loss = tf.reduce_mean(kl_losses)

                autoencoder_loss = image_loss + kl_loss


                # unsupervised_loss_func = VAELossLayer()
                # print('outputs =', outputs)
                # unsupervised_loss_components = unsupervised_loss_func(
                #     self.kl_weight,
                #     self.vae_weight,
                #     posterior_means=outputs['posterior_means'],
                #     posterior_logvars=outputs['posterior_logvars'],
                #     synthetic_image=outputs['final_image_output'],
                #     image=image)
                reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
                #unsupervised_loss = unsupervised_loss_components['loss']
                #log_unsupervised_loss = tf.log(unsupervised_loss)

            if self.has_seg_feature:
                if self.has_autoencoder_feature:
                    if self.net_param.decay > 0.0 and reg_losses:
                        reg_loss = tf.reduce_mean(
                            [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                        loss = seg_loss + autoencoder_loss + reg_loss
                    else:
                        loss = seg_loss + autoencoder_loss
                else:
                    loss = seg_loss
            else:
                if self.has_autoencoder_feature:
                    loss = autoencoder_loss
                else:
                    raise ValueError("Features defined")


            #output collector

            grads = self.optimiser.compute_gradients(
                loss, colocate_gradients_with_ops=True)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])

            # collecting output variables
            limit = 20
            current = 0
            # stack = list()
            # stack.append(TensorElem(image))
            # while not len(stack) == 0 and current < limit:
            #     t = stack[-1]
            #     if t.output is None:
            #         outputs_collector.add_to_collection(
            #             var=tf.reduce_mean(t.tensor),
            #             name="\nmean '{}'".format(t.tensor.name),
            #             average_over_devices=False,
            #             collection=CONSOLE
            #         )
            #         outputs_collector.add_to_collection(
            #             var=tf.reduce_max(t.tensor),
            #             name="\nmax '{}'".format(t.tensor.name),
            #             average_over_devices=False,
            #             collection=CONSOLE
            #         )
            #         t.output = 0
            #         current += 1
            #     else:
            #         if t.output < len(t.tensor.op.outputs):
            #             stack.append(TensorElem(t.tensor.op.outputs[t.output]))
            #             t.output += 1
            #         else:
            #             stack.pop()
            for t in internal:
                outputs_collector.add_to_collection(
                    var=tf.reduce_mean(t),
                    name="\nmean '{}'".format(t.name),
                    average_over_devices=False,
                    collection=CONSOLE)
                outputs_collector.add_to_collection(
                    var=tf.reduce_max(t),
                    name="\nmax '{}'".format(t.name),
                    average_over_devices=False,
                    collection=CONSOLE)

            outputs_collector.add_to_collection(
                var=self.learning_rate, name="learning rate",
                average_over_devices=False, collection=CONSOLE)

            image_min = tf.reduce_min(image)
            image_max = tf.reduce_max(image)
            outputs_collector.add_to_collection(
                var=rescale_image(image, image_min, image_max, 255), name='original image',
                average_over_devices=False, summary_type='image3_axial',
                collection=TF_SUMMARIES)
            outputs_collector.add_to_collection(
                var=data_dict['label_present'][0], name='label_present',
                average_over_devices=False, summary_type='scalar', collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=self.is_validation, name='is_validation',
                average_over_devices=False, summary_type='scalar', collection=CONSOLE)

            if self.has_autoencoder_feature:
                # for g in grads:
                #     outputs_collector.add_to_collection(
                #         var=g[0], name=g[0].name, summary_type='histogram',
                #         collection=TF_SUMMARIES)
                outputs_collector.add_to_collection(
                    var=tf.constant(self.kl_weight), name='kl_weight',
                    average_over_devices=False, summary_type='scalar',
                    collection=TF_SUMMARIES)
                outputs_collector.add_to_collection(
                    var=tf.constant(self.vae_weight), name='vae_weight',
                    average_over_devices=False, summary_type='scalar',
                    collection=TF_SUMMARIES)
                outputs_collector.add_to_collection(
                    var=image_loss, name='image_loss',
                    average_over_devices=False, collection=CONSOLE)
                outputs_collector.add_to_collection(
                    var=kl_loss, name='kl_loss',
                    average_over_devices=False, collection=CONSOLE)
                # outputs_collector.add_to_collection(
                #     var=unsupervised_loss_components['image_max'], name='image_max',
                #     average_over_devices=False, collection=CONSOLE)
                # outputs_collector.add_to_collection(
                #     var=unsupervised_loss_components['synthetic_image_max'], name='synthetic_image_max',
                #     average_over_devices=False, collection=CONSOLE)
                # outputs_collector.add_to_collection(
                #     var=unsupervised_loss_components['kl_loss'], name='kl_loss',
                #     average_over_devices=False, collection=CONSOLE)
                # outputs_collector.add_to_collection(
                #     var=unsupervised_loss_components['l2_loss'], name='l2_loss',
                #     average_over_devices=False, collection=CONSOLE)
                outputs_collector.add_to_collection(
                    var=image_loss, name='image_loss',
                    average_over_devices=False, summary_type='scalar',
                    collection=TF_SUMMARIES)
                outputs_collector.add_to_collection(
                    var=kl_loss, name='kl_loss',
                    average_over_devices=False, summary_type='scalar',
                    collection=TF_SUMMARIES)

                outputs_collector.add_to_collection(
                    var=rescale_image(outputs['image_means'], image_min, image_max, 255),
                    name='synthetic_image_orig_scale',
                    average_over_devices=False, summary_type='image3_axial',
                    collection=TF_SUMMARIES)
                rescaled_min = tf.reduce_min(outputs['image_means'])
                rescaled_max = tf.reduce_max(outputs['image_means'])
                outputs_collector.add_to_collection(
                    var=rescale_image(outputs['image_means'], rescaled_min, rescaled_max, 255),
                    name='synthetic_image',
                    average_over_devices=False, summary_type='image3_axial',
                    collection=TF_SUMMARIES)
                outputs_collector.add_to_collection(
                    var=outputs['image_means'], name='synthetic_image_hist',
                    average_over_devices=False, summary_type='histogram',
                    collection=TF_SUMMARIES)
                outputs_collector.add_to_collection(
                    var=outputs['posterior_means'], name='posterior_means',
                    average_over_devices=False, summary_type='histogram',
                    collection=TF_SUMMARIES)
                outputs_collector.add_to_collection(
                    var=outputs['posterior_logvars'], name='posterior_logvars',
                    average_over_devices=False, summary_type='histogram',
                    collection=TF_SUMMARIES)

            if self.has_seg_feature:
                outputs_collector.add_to_collection(
                    var=seg_loss, name='supervised_loss',
                    average_over_devices=False, collection=CONSOLE)
                outputs_collector.add_to_collection(
                    var=seg_loss, name='supervised_loss',
                    average_over_devices=True, summary_type='scalar',
                    collection=TF_SUMMARIES)

                ground_truth_seg = data_dict.get('label', None)
                outputs_collector.add_to_collection(
                    var=rescale_image(ground_truth_seg, 0, 2, 255), name='ground_truth_seg',
                    average_over_devices=False, summary_type='image3_axial',
                    collection=TF_SUMMARIES)
                outputs_collector.add_to_collection(
                    var=data_dict.get('label', None), name='ground_truth_seg_hist',
                    average_over_devices=False, summary_type='histogram',
                    collection=TF_SUMMARIES)
                predicted_seg = outputs['seg_means']
                float_predicted_seg = tf.cast(predicted_seg, dtype=tf.float32)
                final_pred = tf.nn.softmax(float_predicted_seg)
                outputs_collector.add_to_collection(
                    var=rescale_image(final_pred, 0, 2, 255), name='predicted_seg',
                    average_over_devices=False, summary_type='image3_axial',
                    collection=TF_SUMMARIES)
                outputs_collector.add_to_collection(
                    var=final_pred, name='predicted_seg_hist',
                    average_over_devices=False, summary_type='histogram',
                    collection=TF_SUMMARIES)
                outputs_collector.add_to_collection(
                    var=data_dict['image_location'][0], name='image_index',
                    average_over_devices=False, summary_type='scalar',
                    collection=CONSOLE)
                self.current_id = data_dict['image_location'][0]
        else:
            pass


    def set_iteration_update(self, iteration_message):
        current_iter = iteration_message.current_iter
        if iteration_message.is_training:
            current_lr = pow(self.lr_decay_base, -float(current_iter) / self.lr_decay_rate) * self.base_lr
            iteration_message.data_feed_dict[self.is_validation] = False
        elif iteration_message.is_validation:
            current_lr = self.base_lr
            iteration_message.data_feed_dict[self.is_validation] = True
        iteration_message.data_feed_dict[self.learning_rate] = current_lr


    def interpret_output(self, batch_output):

        if self.is_inference:
            return self.output_decoder.decode_batch(
                batch_output['window'], batch_output['location'])
        return True
