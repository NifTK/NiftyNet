import tensorflow as tf
import niftynet.layer.base_layer as nnlb

class VAELossLayer(nnlb.Layer):
    def __init__(self,
                 name='vae_loss_function'):
        super(VAELossLayer, self).__init__(name=name)

    def layer_op(self,
                 kl_weight=1,
                 l2_weight=1,
                 posterior_means=None,
                 posterior_logvars=None,
                 image=None,
                 synthetic_image=None):
        with tf.device('/cpu:0'):
            print('kl_weight=', kl_weight)
            print('vae_weight=', l2_weight)

            l2_norm_squared = tf.square(image - synthetic_image)
            batch_size = l2_norm_squared.shape[0]
            l2_norm_squared = tf.reshape(l2_norm_squared, shape=[batch_size, -1])
            l2_loss = tf.reduce_mean(l2_norm_squared, axis=[1])

            kl_element = tf.square(posterior_means) + \
                          tf.exp(posterior_logvars) -\
                          posterior_logvars -\
                          1

            kl_loss = tf.reduce_mean(kl_element, axis=[1])


            # return {
            #     'l2_loss': tf.reduce_mean(l2_weight * l2_loss),
            #     'kl_loss': tf.reduce_mean(kl_weight * kl_divergence),
            #     'loss': tf.reduce_mean(l2_weight * l2_loss + kl_weight * kl_divergence),
            #     'image_max': tf.reduce_max(image),
            #     'synthetic_image_max': tf.reduce_max(synthetic_image),
            #     'l2_max': tf.reduce_max(tf.square(synthetic_image - image))
            # }
            return {
                'loss': tf.reduce_mean(l2_loss * l2_weight + kl_loss * kl_weight),
                'l2_loss': tf.reduce_mean(l2_loss * l2_weight),
                'kl_loss': tf.reduce_mean(kl_loss * kl_weight),
                'image_max': tf.reduce_max(image),
                'synthetic_image_max': tf.reduce_max(synthetic_image),
                'l2_max': tf.reduce_max(tf.square(synthetic_image - image))
            }
