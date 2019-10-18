import os

import tensorflow as tf

from niftynet.application.regression_application import \
    RegressionApplication, SUPPORTED_INPUT
from niftynet.engine.sampler_uniform_v2 import UniformSampler
from niftynet.engine.sampler_weighted_v2 import WeightedSampler
from niftynet.engine.application_variables import NETWORK_OUTPUT
from niftynet.io.image_reader import ImageReader
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.layer.pad import PadLayer


class ISampleRegression(RegressionApplication):

    #def initialise_weighted_sampler(self):
    #    if len(self.readers) == 2:
    #        training_sampler = WeightedSampler(
    #            reader=self.readers[0],
    #            window_sizes=self.data_param,
    #            batch_size=self.net_param.batch_size,
    #            windows_per_image=self.action_param.sample_per_volume,
    #            queue_length=self.net_param.queue_length)
    #        validation_sampler = UniformSampler(
    #            reader=self.readers[1],
    #            window_sizes=self.data_param,
    #            batch_size=self.net_param.batch_size,
    #            windows_per_image=self.action_param.sample_per_volume,
    #            queue_length=self.net_param.queue_length)
    #        self.sampler = [[training_sampler, validation_sampler]]
    #    else:
    #        RegressionApplication.initialise_weighted_sampler()


    def initialise_dataset_loader(
            self, data_param=None, task_param=None, data_partitioner=None):
        RegressionApplication.initialise_dataset_loader(
            self, data_param, task_param, data_partitioner)
        if self.is_training:
            return
        if not task_param.error_map:
            # use the regression application implementation
            return

        try:
            reader_phase = self.action_param.dataset_to_infer
        except AttributeError:
            reader_phase = None
        file_lists = data_partitioner.get_file_lists_by(
            phase=reader_phase, action=self.action)
        # modifying the original readers in regression application
        # as we need ground truth labels to generate error maps
        self.readers = [
            ImageReader(['image', 'output']).initialise(
                data_param, task_param, file_list) for file_list in file_lists]

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
            preprocessors.append(PadLayer(
                image_name=SUPPORTED_INPUT,
                border=self.net_param.volume_padding_size))
        self.readers[0].add_preprocessing_layers(preprocessors)

    def connect_data_and_network(self,
                                 outputs_collector=None,
                                 gradients_collector=None):
        if self.is_training:
            # using the original training pipeline
            RegressionApplication.connect_data_and_network(
                self, outputs_collector, gradients_collector)
        else:
            init_aggregator = \
                self.SUPPORTED_SAMPLING[self.net_param.window_sampling][2]
            init_aggregator()

            # modifying the original pipeline so that
            # the error maps are computed instead of the regression output
            with tf.name_scope('validation'):
                data_dict = self.get_sampler()[0][-1].pop_batch_op()
            image = tf.cast(data_dict['image'], tf.float32)
            net_out = self.net(image, is_training=self.is_training)

            if self.regression_param.error_map:
                # writing error maps to folder without prefix
                error_map_folder = os.path.join(
                    os.path.dirname(self.action_param.save_seg_dir),
                    'error_maps')
                self.output_decoder.output_path = error_map_folder
                self.output_decoder.prefix = ''

                # computes absolute error
                target = tf.cast(data_dict['output'], tf.float32)
                net_out = tf.squared_difference(target, net_out)

            # window output and locations for aggregating volume results
            outputs_collector.add_to_collection(
                var=net_out, name='window',
                average_over_devices=False, collection=NETWORK_OUTPUT)
            outputs_collector.add_to_collection(
                var=data_dict['image_location'], name='location',
                average_over_devices=False, collection=NETWORK_OUTPUT)
