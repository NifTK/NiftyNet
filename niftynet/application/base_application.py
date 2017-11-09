from six import with_metaclass
from niftynet.utilities import util_common
from niftynet.layer.base_layer import TrainableLayer
import tensorflow as tf

class SingletonApplication(type):
    _instances = None

    def __call__(cls, *args, **kwargs):
        if cls._instances is None:
            cls._instances = \
                super(SingletonApplication, cls).__call__(*args, **kwargs)
        # else:
        #     raise RuntimeError('application instance already started.')
        return cls._instances


class BaseApplication(with_metaclass(SingletonApplication, object)):
    """
    BaseApplication represents an interface.
    Each application type_str should support to use
    the standard training and inference driver
    """
    REQUIRED_CONFIG_SECTION = None

    is_training = True

    # input of the network
    readers = None
    sampler = None

    # the network
    net = None

    # training the network
    optimiser = None
    gradient_op = None

    # interpret network output
    output_decoder = None

    def check_initialisations(self):
        if self.readers is None:
            raise NotImplementedError('reader should be initialised')
        if self.sampler is None:
            raise NotImplementedError('sampler should be initialised')
        if self.net is None:
            raise NotImplementedError('net should be initialised')
        if not isinstance(self.net, TrainableLayer):
            raise ValueError('self.net should be an instance'
                             ' of niftynet.layer.TrainableLayer')
        if self.optimiser is None and self.is_training:
            raise NotImplementedError('optimiser should be initialised')
        if self.gradient_op is None and self.is_training:
            raise NotImplementedError('gradient_op should be initialised')
        if self.output_decoder is None and not self.is_training:
            raise NotImplementedError('output decoder should be initialised')

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        """
        this function initialise self.readers

        :param data_param: input modality specifications
        :param task_param: contains task keywords for grouping data_param
        :param data_partitioner:
                           specifies train/valid/infer splitting if needed
        :return:
        """
        raise NotImplementedError

    def initialise_sampler(self):
        """
        set samplers take self.reader as input and generates
        sequences of ImageWindow that will be fed to the networks
        This function sets self.sampler
        """
        raise NotImplementedError

    def initialise_network(self):
        """
        This function create an instance of network
        sets self.net
        :return: None
        """
        self.is_validation = tf.placeholder_with_default(False, [],
                                                         'is_validation')

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        raise NotImplementedError

    def interpret_output(self, batch_output):
        """
        implement output interpretations, e.g., save to hard drive
        cache output windows
        :param batch_output: outputs by running the tf graph
        :return: True indicates the driver should continue the loop
                 False indicates the drive should stop
        """
        raise NotImplementedError

    def set_network_update_op(self, gradients):
        grad_list_depth = util_common.list_depth_count(gradients)
        if grad_list_depth == 3:
            # nested depth 3 means: gradients list is nested in terms of:
            # list of networks -> list of network variables
            self.gradient_op = [self.optimiser.apply_gradients(grad)
                                for grad in gradients]
        elif grad_list_depth == 2:
            # nested depth 2 means:
            # gradients list is a list of variables
            self.gradient_op = self.optimiser.apply_gradients(gradients)
        else:
            raise NotImplementedError(
                'This app supports updating a network, or list of networks')

    def stop(self):
        for sampler_set in self.get_sampler():
            for sampler in sampler_set:
                if sampler:
                    sampler.close_all()

    def iter_ops(self, start_iter=0, end_iter=1):
        param = self.action_param
        training_op_generator = self.training_ops(start_iter=start_iter,
                                                  end_iter=end_iter)
        for iter_i, op, feed_dict in training_op_generator:
            if self.has_validation_data and param.validate_every_n > 0 and \
                            iter_i % param.validate_every_n == 0:
                feed_dict_validation = feed_dict.copy()
                feed_dict_validation[self.is_validation]=True
                for iter_j in range(param.validation_iters):
                    yield 'Validate', iter_i, False, 2, \
                          self.is_validation, feed_dict_validation

            feed_dict[self.is_validation]=False
            save = param.save_every_n > 0 and \
                   iter_i % param.save_every_n == 0
            save_log = 1 if (param.tensorboard_every_n > 0 and \
                       iter_i % param.tensorboard_every_n == 0) else 0
            yield 'Train', iter_i, save, save_log, self.gradient_op, feed_dict

    def training_ops(self, start_iter=0, end_iter=1):
        """
        Specify the network update operation at each iteration
        app can override this updating method if necessary
        """
        end_iter = max(start_iter, end_iter)
        for iter_i in range(start_iter, end_iter):
            yield iter_i, self.gradient_op, {}

    def get_sampler(self):
        return self.sampler
