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
        self.app = None
        self.graph = None
        self.saver = None

        self.is_training = True
        self.num_threads = 0
        self.num_gpus = 0

        self.model_dir = None
        self.max_checkpoints = 20
        self.save_every_n = 10
        self.initial_iter = 0
        self.final_iter = 0

        self._init_op = None

    def initialise_application(self, csv_dict, param):
        self.is_training = (param.action == "train")

        # hardware-related parameters
        ApplicationDriver._set_cuda_device(param.cuda_devices)
        self.num_threads = max(param.num_threads, 1)
        self.num_gpus = param.num_gpus
        self.max_checkpoints = param.max_checkpoints
        self.save_every_n = param.save_every_n
        self.model_dir = ApplicationDriver._touch_folder(param.model_dir)

        # create an application and assign user-specified parameters
        self.app = ApplicationDriver._create_app(param.application_type)
        self.app.set_param(param)

        # initialise data input, and the tf graph
        self.app.initialise_dataset_loader(csv_dict)
        self.graph = self._create_graph()

        self.initial_iter = param.starting_iter \
            if self.is_training else param.inference_iter
        self.final_iter = param.max_iter

    def run_application(self):
        assert self.graph is not None, \
            "please call initialise_application first"
        config = ApplicationDriver._tf_config()
        with tf.Session(config=config, graph=self.graph) as session:
            if self.is_training:
                self._training_loop(session)
            else:
                self._inference_loop(session)

    def _randomly_init_or_restore_variables(self, sess):
        if self.is_training and self.initial_iter == 0:
            sess.run(self._init_op)
            print('trainable parameters from random initialisations ...')
            return
        assert os.path.exists(self.model_dir), \
            "Model folder not found {}, please check" \
            "config parameter: model_dir".format(self.model_dir)
        checkpoint = os.path.join(
            self.model_dir, 'model.ckpt-{}'.format(self.initial_iter))
        assert tf.train.get_checkpoint_state(self.model_dir) is not None, \
            "Model file not found {}*, please check" \
            "config parameter: model_dir and *_iter".format(checkpoint)
        print('Restore parameters from {} ...'.format(checkpoint))
        self.saver.restore(sess, checkpoint)
        return

    def _training_loop(self, sess):
        self._randomly_init_or_restore_variables(sess)

        coord = tf.train.Coordinator()
        self.app.get_sampler().run_threads(sess, coord, self.num_threads)
        for (iter_i, train_op) in \
                self.app.get_iterative_op(self.initial_iter, self.final_iter):
            #save_path = os.path.join(self.model_dir, 'model.ckpt')
            #self.saver.save(sess, save_path, global_step=iter_i)
            #import pdb; pdb.set_trace()
            if coord.should_stop():
                break
            print(iter_i)
            sess.run(train_op)

            if iter_i % self.save_every_n == 0 and iter_i > 0:
                save_path = os.path.join(self.model_dir, 'model.ckpt')
                self.saver.save(sess, save_path, global_step=iter_i)
                print('Iter {} model saved at {}'.format(iter_i, save_path))

    def _inference_loop(self, sess):
        pass

    def _create_graph(self):
        graph = tf.Graph()
        main_device = self._device_string(0, is_worker=False)
        # start constructing the graph, handling training and inference cases
        with graph.as_default(), tf.device(main_device):
            # initialise sampler and network, these are connected in
            # the context of multiple gpus
            self.app.initialise_sampler(is_training=self.is_training)
            self.app.initialise_network()

            # defining and collecting variables from multiple gpus
            net_outputs = []
            training_grads = [] if self.is_training else None
            bn_ops = None
            for gpu_id in range(0, max(self.num_gpus, 1)):
                worker_device = self._device_string(gpu_id, is_worker=True)
                with tf.device(worker_device):
                    # compute gradients for one device of multiple device
                    # data parallelism
                    output = self.app.connect_data_and_network(
                        self.is_training, training_grads)
                    net_outputs.append(output)
                    if gpu_id == 0 and self.is_training:
                        # batch normalisation updates from 1st device only
                        bn_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            # assigning output variables back to each application
            self.app.set_output_op(net_outputs)

            # moving average operation
            variable_averages = tf.train.ExponentialMovingAverage(0.9)
            trainables = tf.trainable_variables()
            moving_ave_op = variable_averages.apply(trainables)

            # training operation
            if self.is_training:
                updates_op = [moving_ave_op]
                updates_op.extend(bn_ops) if bn_ops is not None else None
                with graph.control_dependencies(updates_op):
                    averaged_grads = util.average_gradients(training_grads)
                    self.app.create_network_update_op(averaged_grads)

            # initialisation operation
            self._init_op = tf.global_variables_initializer()

            # saving operation
            self.saver = tf.train.Saver(max_to_keep=self.max_checkpoints)

        # no more operation definitions after this point
        tf.Graph.finalize(graph)
        return graph

    @staticmethod
    def _create_app(app_type_string):
        _app_module = ApplicationFactory.import_module(app_type_string)
        return _app_module()

    @staticmethod
    def _tf_config():
        config = tf.ConfigProto()
        config.log_device_placement = False
        config.allow_soft_placement = True
        return config

    @staticmethod
    def _touch_folder(model_dir):
        model_dir = os.path.join(model_dir, 'models')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        absolute_dir = os.path.abspath(model_dir)
        print('accessing output folder: {}'.format(absolute_dir))
        return absolute_dir

    @staticmethod
    def _set_cuda_device(cuda_devices):
        # TODO: refactor this OS-denpendent function
        if not (cuda_devices == '""'):
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
            print("set CUDA_VISIBLE_DEVICES to {}".format(cuda_devices))
        else:
            # using Tensorflow default choice
            pass

    def _device_string(self, id=0, is_worker=True):
        if self.num_gpus <= 0:  # user specified no gpu at all
            return '/cpu:{}'.format(id)
        if self.is_training:
            device = 'gpu' if is_worker else 'cpu'
            return '/{}:{}'.format(device, id)
        else:
            return '/gpu:0'  # always use one GPU for inference
