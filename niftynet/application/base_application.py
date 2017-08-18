from niftynet.utilities import util_common


class SingletonApplication(type):
    _instances = None

    def __call__(cls, *args, **kwargs):
        if cls._instances is None:
            cls._instances = \
                super(SingletonApplication, cls).__call__(*args, **kwargs)
        # else:
        #     raise RuntimeError('application instance already started.')
        return cls._instances


class BaseApplication(object):
    """
    BaseApplication represents an interface.
    Each application type should support to use
    the standard training and inference driver
    """
    __metaclass__ = SingletonApplication

    is_training = True
    reader = None
    sampler = None
    net = None

    optimiser = None
    gradient_op = None

    output_decoder = None

    def check_initialisations(self):
        if self.reader is None:
            raise NotImplementedError('reader should be initialised')
        if self.sampler is None:
            raise NotImplementedError('sampler should be initialised')
        if self.net is None:
            raise NotImplementedError('net should be initialised')
        if self.optimiser is None and self.is_training:
            raise NotImplementedError('optimiser should be initialised')
        if self.gradient_op is None and self.is_training:
            raise NotImplementedError('gradient_op should be initialised')
        if self.output_decoder is None and not self.is_training:
            raise NotImplementedError('output decoder should be initialised')

    def initialise_dataset_loader(self, data_param=None, task_param=None):
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
        raise NotImplementedError

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        raise NotImplementedError

    def interpret_output(self, batch_output):
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
        for sampler in self.get_sampler():
            if sampler:
                sampler.close_all()

    def training_ops(self, start_iter=0, end_iter=1):
        end_iter = max(start_iter, end_iter)
        for iter_i in range(start_iter, end_iter):
            yield iter_i, self.gradient_op

    def get_sampler(self):
        return self.sampler
