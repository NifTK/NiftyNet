import tensorflow as tf

import os
from niftynet.application.regression_application import RegressionApplication
from niftynet.engine.application_variables import NETWORK_OUTPUT
from niftynet.io.image_reader import ImageReader
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.pad import PadLayer

SUPPORTED_INPUT = set(['image', 'output', 'weight', 'sampler'])


class ISampleRegression(RegressionApplication):
    REQUIRED_CONFIG_SECTION = "REGRESSION"

    def __init__(self, net_param, action_param, is_training):
        RegressionApplication.__init__(
            self, net_param, action_param, is_training)

    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        RegressionApplication.initialise_dataset_loader(
            self, data_param, task_param, data_partitioner)

        if self.is_training:
            return

        if not task_param.error_map:
            return
        # generating error_map, so replacing the reader using training data
        self.readers[0] = ImageReader(['image'])
        self.readers[0].initialise(
            data_param, task_param, data_partitioner.train_files)

        mean_var_normaliser = MeanVarNormalisationLayer(image_name='image')
        histogram_normaliser = None
        if self.net_param.histogram_ref_file:
            histogram_normaliser = HistogramNormalisationLayer(
                image_name='image',
                modalities=vars(task_param).get('image'),
                model_filename=self.net_param.histogram_ref_file,
                norm_type=self.net_param.norm_type,
                cutoff=self.net_param.cutoff,
                name='hist_norm_layer')

        preprocessors = []
        if self.net_param.normalisation:
            preprocessors.append(histogram_normaliser)
        if self.net_param.whitening:
            preprocessors.append(mean_var_normaliser)

        if self.net_param.volume_padding_size:
            preprocessors.append(
                PadLayer(image_name=SUPPORTED_INPUT,
                         border=self.net_param.volume_padding_size))

        self.readers[0].add_preprocessing_layers(preprocessors)


    def initialise_network(self):
        RegressionApplication.initialise_network(self)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):

        if self.is_training:
            RegressionApplication.connect_data_and_network(
                self, outputs_collector, gradients_collector)
        else:
            with tf.name_scope('validation'):
                data_dict = self.get_sampler()[0][-1].pop_batch_op()
            image = tf.cast(data_dict['image'], tf.float32)
            net_out = self.net(image, is_training=self.is_training)

            if self.regression_param.error_map:
                # replace the output dir to a `error_maps` folder
                error_map_folder = os.path.join(
                    os.path.dirname(self.action_param.save_seg_dir),
                    'error_maps')
                self.action_param.save_seg_dir = error_map_folder
                # computes absolute error
                errors = tf.abs(image - net_out)
                outputs_collector.add_to_collection(
                    var=errors, name='window',
                    average_over_devices=False, collection=NETWORK_OUTPUT)
            else:
                outputs_collector.add_to_collection(
                    var=net_out, name='window',
                    average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            init_aggregator = \
                self.SUPPORTED_SAMPLING[self.net_param.window_sampling][2]
            init_aggregator()


    def interpret_output(self, batch_output):
        if not self.is_training:
            return self.output_decoder.decode_batch(
                batch_output['window'], batch_output['location'])
        else:
            return True
