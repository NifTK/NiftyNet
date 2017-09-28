import tensorflow as tf

from niftynet.application.segmentation_application import \
    SegmentationApplication
from niftynet.engine.application_factory import OptimiserFactory
from niftynet.engine.application_variables import CONSOLE
from niftynet.engine.application_variables import TF_SUMMARIES
from niftynet.layer.loss_segmentation import LossFunction

SUPPORTED_INPUT = {'image', 'label', 'weight'}


class DecayLearningRateApplication(SegmentationApplication):
    REQUIRED_CONFIG_SECTION = "SEGMENTATION"

    def __init__(self, net_param, action_param, is_training):
        SegmentationApplication.__init__(
            self, net_param, action_param, is_training)
        tf.logging.info('starting decay learning segmentation application')
        self.learning_rate = None

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        data_dict = self.get_sampler()[0].pop_batch_op()
        image = tf.cast(data_dict['image'], tf.float32)
        net_out = self.net(image, self.is_training)

        if self.is_training:
            with tf.name_scope('Optimiser'):
                self.learning_rate = tf.placeholder(tf.float32, shape=[])
                optimiser_class = OptimiserFactory.create(
                    name=self.action_param.optimiser)
                self.optimiser = optimiser_class.get_instance(
                    learning_rate=self.learning_rate)
            loss_func = LossFunction(
                n_class=self.segmentation_param.num_classes,
                loss_type=self.action_param.loss_type)
            data_loss = loss_func(
                prediction=net_out,
                ground_truth=data_dict.get('label', None),
                weight_map=data_dict.get('weight', None))
            if self.net_param.decay > 0.0:
                reg_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                if reg_losses:
                    reg_loss = tf.reduce_mean(
                        [tf.reduce_mean(reg_loss) for reg_loss in reg_losses])
                    loss = data_loss + reg_loss
            else:
                loss = data_loss
            grads = self.optimiser.compute_gradients(loss)
            # collecting gradients variables
            gradients_collector.add_to_collection([grads])
            # collecting output variables
            outputs_collector.add_to_collection(
                var=data_loss, name='dice_loss',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=self.learning_rate, name='lr',
                average_over_devices=False, collection=CONSOLE)
            outputs_collector.add_to_collection(
                var=data_loss, name='dice_loss',
                average_over_devices=True, summary_type='scalar',
                collection=TF_SUMMARIES)
        else:
            # converting logits into final output for
            # classification probabilities or argmax classification labels
            SegmentationApplication.connect_data_and_network(
                self, outputs_collector, gradients_collector)

    def training_ops(self, start_iter=0, end_iter=1):
        if start_iter > end_iter:
            start_iter, end_iter = end_iter, start_iter
        current_lr = self.action_param.lr
        for iter_i in range(start_iter, end_iter):
            if iter_i % 3 == 0 and iter_i > 0:
                # halved every 3 iteration
                current_lr = current_lr / 2.0
            yield iter_i, self.gradient_op[0], {self.learning_rate: current_lr}
