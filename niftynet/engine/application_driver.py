import os

import tensorflow as tf

from niftynet.utilities import misc_common as util


class ApplicationFactory(object):
    from niftynet.application.segmentation_application import \
        SegmentationApplication
    from niftynet.application.autoencoder_application import \
        AutoencoderApplication
    from niftynet.application.gan_application import GANApplication

    application_dict = {'segmentation': SegmentationApplication,
                        'autoencoder': AutoencoderApplication,
                        'gan': GANApplication}

    @staticmethod
    def import_module(type_string):
        return ApplicationFactory.application_dict[type_string]


class ApplicationDriver(object):
    def __init__(self):
        self._app_graph = None
        self.is_training = True
        self.app = None
        self.num_threads = 0
        self.num_gpus = 0

        self._init_op = None
        self.max_checkpoints = 20

    def initialise_application(self, csv_dict, param):
        self.is_training = param.action == "train"

        # hardware-related parameters
        self.num_threads = max(param.num_threads, 1)
        self.num_gpus = param.num_gpus
        if not (param.cuda_devices == '""'):
            os.environ["CUDA_VISIBLE_DEVICES"] = param.cuda_devices
            print("set CUDA_VISIBLE_DEVICES to {}".format(param.cuda_devices))

        self.max_checkpoints = param.max_checkpoints

        # create an application and assign user-specified parameters
        self.app = self._create_application_instance(param.application_type)
        self.app.set_param(param)

        # initialise data input, and the tf graph
        self.app.initialise_dataset_loader(csv_dict)
        self._app_graph = self._create_graph()

    def run_application(self):
        assert self._app_graph is not None,\
            "please call initialise_application first"

        config = tf.ConfigProto()
        config.log_device_placement = False
        config.allow_soft_placement = True

        with tf.Session(config=config, graph=self._app_graph) as sess:
            sess.run(self._init_op)
            coord = tf.train.Coordinator()
            self.app.get_sampler().run_threads(sess, coord, self.num_threads)
            for iter_i, app_op in self.app.get_iterative_op(0, 1):
                if coord.should_stop():
                    break
                output = sess.run(app_op)

    def _create_application_instance(self, app_type_string):
        self._app_module = ApplicationFactory.import_module(app_type_string)
        return self._app_module()

    def _create_graph(self):
        graph = tf.Graph()
        main_device = self._device_string(0, self.is_training, False)
        with graph.as_default(), tf.device(main_device):
            # initialise sampler and network, these are connected in
            # the context of multiple gpus
            self.app.initialise_sampler(is_training=self.is_training)
            self.app.initialise_network()

            training_grads = [] if self.is_training else None
            net_outputs = []
            for gpu_id in range(0, max(self.num_gpus, 1)):
                worker_device = self._device_string(gpu_id, self.is_training)
                with tf.device(worker_device):
                    # compute gradients for one device of multiple device
                    # data parallelism
                    output = self.app.connect_data_and_network(
                        self.is_training, training_grads)
                    net_outputs.append(output)
            self.app.set_output_op(net_outputs)
            if self.is_training:
                averaged_grads = util.average_gradients(training_grads)
                self.app.set_gradients_op(averaged_grads)

            self._init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep=self.max_checkpoints)
        return graph

    def _device_string(self, id=0, is_training=False, is_worker=True):
        if self.num_gpus <= 0:  # user specified no gpu at all
            return '/cpu:{}'.format(id)
        if is_training:
            if is_worker:
                return '/gpu:{}'.format(id)
            else:
                return '/cpu:{}'.format(id)
        if not is_training:
            return '/gpu:0'  # always try GPU for inference
        return '/cpu:{}'.format(id)
