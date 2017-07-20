import os

import tensorflow as tf


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
    def __init__(self, param, is_training):
        self.param = param
        self.is_training = is_training
        self.app = None
        self._app_graph = None
        if not (param.cuda_devices == '""'):
            os.environ["CUDA_VISIBLE_DEVICES"] = param.cuda_devices
            print("set CUDA_VISIBLE_DEVICES to {}".format(param.cuda_devices))

    def _get_device_string(self,
                           id=0,
                           num_gpus=0,
                           is_training=False,
                           is_worker=False):
        if is_training and is_worker and num_gpus > 0:
            return '/gpu:{}'.format(id)
        if not is_training and num_gpus > 0:
            return '/gpu:{}'.format(id)
        return '/cpu:{}'.format(id)

    def _create_application_instance(self, app_type_string):
        _app_module = ApplicationFactory.import_module(app_type_string)
        return _app_module()

    def _create_graph(self):
        graph = tf.Graph()
        device_string = self._get_device_string(id=0,
                                                num_gpus=self.param.num_gpus,
                                                is_training=self.is_training,
                                                is_worker=False)
        with graph.as_default(), tf.device(device_string):
            self.app.initialise_sampler(is_training=self.is_training)
            self.app.initialise_network()
            for gpu_id in range(0, self.param.num_gpus):
                device_str = self._get_device_string(
                    id=gpu_id,
                    num_gpus=self.param.num_gpus,
                    is_training=self.is_training,
                    is_worker=True)
                with tf.device(device_str):
                    self.app._connect_data_and_network(self.is_training, gpu_id)
        return graph

    def initialise_application(self, csv_dict):
        self.app = self._create_application_instance(
            self.param.application_type)
        self.app.set_param(self.param)
        self.app.initialise_dataset_loader(csv_dict)
        self._app_graph = self._create_graph()

    def run(self):
        if self._app_graph is None:
            return
        #
        #         if is_training:
        #             model_ops = self.app.training_gradient_ops(network_outputs)
        #         else:
        #             model_ops = self.app.inference_ops(outputs)
        #         log_ops = self.app.log_ops()
        #         saver = ...
        #
        #     with tf.Session(config=config, graph=graph) as sess:
        #
        #         self.randomly_initialise_or_restore(sess, saver)
        #         coord = tf.train.Coordinator()
        #         writer = tf.summary.FileWriter()
        #         input_buffers.run_threads(sess, coord, param.num_threads)
        #
        #         for i in iterations:
        #             if coord.should_stop():
        #                 break
        #             model_output = sess.run(model_ops)
        #             logs = sess.run(log_ops)
        #             self.app.dataset_recontruction_from(model_output)
        #

