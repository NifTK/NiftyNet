import tensorflow as tf

from niftynet.layer.base_layer import TrainableLayer
from niftynet.layer.fully_connected import FullyConnectedLayer


class GaussianSampler(TrainableLayer):
    """
        This predicts the mean and logvariance parameters,
        then generates an approximate sample from the posterior.
    """

    def __init__(self,
                 number_of_latent_variables,
                 number_of_samples_from_posterior,
                 logvars_upper_bound,
                 logvars_lower_bound,
                 w_initializer=None,
                 w_regularizer=None,
                 b_initializer=None,
                 b_regularizer=None,
                 name='gaussian_sampler'):

        super(GaussianSampler, self).__init__(name=name)

        self.number_of_latent_variables = number_of_latent_variables
        self.number_of_samples = number_of_samples_from_posterior
        self.logvars_upper_bound = logvars_upper_bound
        self.logvars_lower_bound = logvars_lower_bound

        self.initializers = {'w': w_initializer, 'b': b_initializer}
        self.regularizers = {'w': w_regularizer, 'b': b_regularizer}

    def layer_op(self, codes, is_training):

        def clip(input):
            # This is for clipping logvars,
            # so that variances = exp(logvars) behaves well
            output = tf.maximum(input, self.logvars_lower_bound)
            output = tf.minimum(output, self.logvars_upper_bound)
            return output

        encoder_means = FullyConnectedLayer(
            n_output_chns=self.number_of_latent_variables,
            with_bn=False,
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='encoder_fc_means_{}'.format(self.number_of_latent_variables))
        print(encoder_means)

        encoder_logvars = FullyConnectedLayer(
            n_output_chns=self.number_of_latent_variables,
            with_bn=False,
            acti_func=None,
            w_initializer=self.initializers['w'],
            w_regularizer=self.regularizers['w'],
            name='encoder_fc_logvars_{}'.format(
                self.number_of_latent_variables))
        print(encoder_logvars)

        # Predict the posterior distribution's parameters
        posterior_means = encoder_means(codes, is_training)
        posterior_logvars = clip(encoder_logvars(codes, is_training))

        if self.number_of_samples == 1:
            noise_sample = tf.random_normal(tf.shape(posterior_means),
                                            0.0,
                                            1.0)
        else:
            expanded_shape = (self.number_of_samples,) + tuple(map(lambda i: i.value, posterior_means.shape[1:]))
            noise_sample = tf.random_normal(expanded_shape, 0.0, 1.0)
            # sample_shape = tf.concat(
            #     [tf.constant(self.number_of_samples, shape=[1, ]),
            #      tf.shape(posterior_means)], axis=0)
            # noise_sample = tf.reduce_mean(
            #     tf.random_normal(sample_shape, 0.0, 1.0), axis=0)

        return [
            posterior_means + tf.exp(0.5 * posterior_logvars) * noise_sample,
            posterior_means, posterior_logvars]