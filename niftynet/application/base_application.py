import abc

import tensorflow as tf

import niftynet.engine


class SingletonApplication(abc.ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # only allow one application instance
            cls._instances[cls] = \
                super(SingletonApplication, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class BaseApplication(object):
    """ BaseApplication represents an interface that each application type
            should support to use the standard training and inference engine
    """
    __metaclass__ = SingletonApplication

    # @abc.abstractmethod
    # def inference_sampler(self):
    #     """
    #     Returns a Sampler used by DeployInputBuffer, the inference input buffer
    #     This defines the ordered sequence of inputs that will be fed to the engine
    #     """
    #     pass

    @abc.abstractmethod
    def initialise_sampler(self, is_training):
        """
        Returns a Sampler used by TrainEvalInputBuffer, the training input buffer
        This defines the sequence of inputs that will be fed to the engine
        """
        pass

    @abc.abstractmethod
    def initialise_network(self, train_dict, is_training):
        """
        This method returns the network output ops for the training network.  This typically
        involves instantiating the net_class and calling it with some or all of the fields
        in the dictionary created by the sampler.

        Parameters:
        train_dict: a dictionary of tensors as given by the sampler
        is_training: a boolean that is True in training and False in inference

        Returns a list of the output tensors from the network
        """
        pass

    @abc.abstractmethod
    def net_inference(self, train_dict, is_training):
        """
        This method returns the network output ops for the inference network.    This typically
        involves instantiating the net_class and calling it with some or all of the fields
        in the dictionary created by the sampler, and optionally doing any tensorflow-based
        post-processing. If train_dict contains information needed for inference, it can be
        added to the network outputs here.

        Parameters:
        train_dict: a dictionary of tensors as given by the sampler
        is_training: a boolean that is True in training and False in inference

        Returns a list of the output tensors from the network
        """
        return self._net(train_dict['images'], is_training), train_dict['info']

    @abc.abstractmethod
    def loss_func(self, train_dict, net_outputs):
        pass

    def train(self, train_dict):
        """
        Returns a list of possible compute_gradients ops to be run each
        training iteration. Default implementation returns gradients for all
        variables from one Adam optimizer
        """
        # optimizer
        with tf.name_scope('Optimizer'):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self._param.lr)
        net_outputs = self.net(train_dict, is_training=True)
        with tf.name_scope('Loss'):
            loss = self.loss_func(train_dict, net_outputs)
            if self._param.decay > 0:
                reg_losses = tf.get_collection(
                    tf.GraphKeys.REGULARIZATION_LOSSES)
                reg_loss = tf.reduce_mean([tf.reduce_mean(reg_loss)
                                           for reg_loss in reg_losses])
                loss = loss + reg_loss

        # Averages are in name_scope for Tensorboard naming; summaries are outside for console naming
        logs = [['loss', loss]]
        with tf.name_scope('ConsoleLogging'):
            logs += self.logs(train_dict, net_outputs)
        for tag, val in logs:
            tf.summary.scalar(tag, val, [niftynet.engine.logging.CONSOLE,
                                         niftynet.engine.logging.LOG])
        with tf.name_scope('ComputeGradients'):
            grads = [self.optimizer.compute_gradients(loss)]
        # add compute gradients ops for each type of optimizer_op
        return grads

    def regularizers(self):
        w_regularizer = None
        b_regularizer = None
        if self._param.reg_type.lower() == 'l2':
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l2_regularizer(self._param.decay)
            b_regularizer = regularizers.l2_regularizer(self._param.decay)
        elif self._param.reg_type.lower() == 'l1':
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l1_regularizer(self._param.decay)
            b_regularizer = regularizers.l1_regularizer(self._param.decay)
        return w_regularizer, b_regularizer

    @abc.abstractmethod
    def inference_loop(self, sess, coord, net_out):
        pass

    def train_op_generator(self, apply_ops):
        """
        Returns a generator defining the sequence of optimization ops to run. These must
        be selected from apply_ops, which corresponds one-to-one with the list of ops
        returned by gradient_ops. Default behaviour is to run all apply_ops in sequence
        Parameters
        apply_ops: list of tensor_ops in the same order as the corresponding list of ops from
                             self.optimizer_ops
        """
        while True:
            yield apply_ops
