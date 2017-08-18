import abc

from niftynet.utilities import util_common


class SingletonApplication(abc.ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # only allow one application instance
            cls._instances[cls] = \
                super(SingletonApplication, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class BaseApplication(object):
    """
    BaseApplication represents an interface.
    Each application type should support to use
    the standard training and inference driver
    """
    __metaclass__ = SingletonApplication

    def __init__(self):
        self._reader = None
        self._sampler = None
        self._net = None

        self._gradient_op = None

    def get_sampler(self):
        return self._sampler

    @abc.abstractmethod
    def initialise_dataset_loader(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def initialise_sampler(self):
        """
        set samplers take self._reader as input and generates
        sequences of ImageWindow that will be fed to the networks
        This function sets self._sampler
        """
        pass

    @abc.abstractmethod
    def initialise_network(self):
        """
        This function create an instance of network
        sets self._net
        :return: None
        """
        pass

    @abc.abstractmethod
    def connect_data_and_network(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def interpret_output(self, *args, **kwargs):
        pass

    def set_network_update_op(self, gradients):
        grad_list_depth = util_common.list_depth_count(gradients)
        if grad_list_depth == 3:
            # nested depth 3 means: gradients list is nested in terms of:
            # list of networks -> list of network variables
            self._gradient_op = [self.optimizer.apply_gradients(grad)
                                 for grad in gradients]
        elif grad_list_depth == 2:
            # nested depth 2 means:
            # gradients list is a list of variables
            self._gradient_op = self.optimizer.apply_gradients(gradients)
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
            yield iter_i, self._gradient_op
