# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

from niftynet.layer.base_layer import TrainableLayer


class GenericGAN(TrainableLayer):
  """
    ### Description
    Generic Generative Adversarial Network

    ### Diagram

    RANDOM NOISE --> [GENERATOR] --> [DISCRIMINATOR] --> fake logits
    TRAINING SET ------------------> [DISCRIMINATOR] --> real logits

    ### Constraints

  """
  def __init__(self, generator, discriminator, name='generic_GAN'):
    self._generator=generator
    self._discriminator=discriminator
    super(GenericGAN, self).__init__(name=name)
  def layer_op(self, random_source, population,is_training):
    image=self._generator(random_source,population.shape.as_list()[1:],is_training)
    fake_logits = self._discriminator(image,is_training)
    real_logits = self._discriminator(population/2,is_training)
    diff=image-population/2
    #logging.image3_axial('fake',(image+1)*127,1,[logging.LOG])
    #logging.image3_axial('real',tf.maximum(0.,tf.minimum(255.,(population/2+1)*127)),1,[logging.LOG])
    return image, real_logits, fake_logits, diff

class SimpleGAN(GenericGAN):
  """
    ### Description
    Specification of generator and discriminator for generic gan

    ### Building blocks
    [GENERATOR]       - See ImageGenerator class below
    [DISCRIMINATOR]   - See ImageDiscriminator class below

    ### Diagram

    RANDOM NOISE --> [GENERATOR] --> [DISCRIMINATOR] --> fake logits
    TRAINING SET ------------------> [DISCRIMINATOR] --> real logits

    ### Constraints

  """
  def __init__(self, name='simple_GAN'):
    generator=ImageGenerator(hidden_layer_channels=[128,64,32,16],
                             name='generator')
    discriminator = ImageDiscriminator(hidden_layer_channels=[16,32,64,128],name='discriminator')
    super(SimpleGAN, self).__init__(generator, discriminator, name)

class ImageGenerator(TrainableLayer):
  """
    ### Description

    ### Diagram

    ### Constraints
  """
  def __init__(self, hidden_layer_channels, name):
    """

    :param hidden_layer_channels:
    :param name: layer name
    """
    super(ImageGenerator, self).__init__(name=name)
    self._num_layers = len(hidden_layer_channels)
    self._layer_channels = hidden_layer_channels
    self.initializers = {'w':tf.contrib.layers.variance_scaling_initializer(),'b':tf.constant_initializer(0)}

  def layer_op(self, random_source, image_size,is_training):
    """

    :param random_source: tensor, random noise to start generation
    :param image_size: output image size
    :param is_training: boolean, True if network is in training mode
    :return: tensor, generated image
    """
    spatial_rank = len(image_size)-1
    batch_size = random_source.shape.as_list()[0]
    noise_size=random_source.shape.as_list()[1]
    intermediate_sizes = [[]]*(self._num_layers)+[image_size]
    for it in range(self._num_layers,0,-1):
      intermediate_sizes[it-1]= [int(round(i/2)) for i in intermediate_sizes[it][:-1]]+[self._layer_channels[it-1]]

    # Define first kernel noise->image
    noise_to_image_kernel = tf.get_variable("G_fcW1", shape=[1,np.prod(intermediate_sizes[0]),noise_size], initializer=self.initializers['w'])
    noise_to_image_bias = tf.get_variable("G_fcb1", shape=[1,np.prod(intermediate_sizes[0]),1], initializer=self.initializers['b'])
    image = tf.reshape(tf.matmul(tf.tile(noise_to_image_kernel,[batch_size,1,1]),tf.expand_dims(random_source,2))+noise_to_image_bias,[batch_size]+intermediate_sizes[0])
    # define components of upsampling units
    acti_func=tf.nn.relu
    dropout_func = lambda x: tf.nn.dropout(x,.5)
    #dropout_func = tf.identity
    norm_func = tf.contrib.layers.batch_norm
    upscale=['resize','conv_transpose'][0]
    if spatial_rank==2:
      conv_func=tf.nn.conv2d
      if upscale=='conv_transpose':
        conv_t_func=tf.nn.conv2d_transpose
      else:
        conv_t_func=lambda x, Wt, sz, st, p: conv_func(tf.image.resize_images(x,sz[1:3]),tf.transpose(Wt,[0,1,3,2]),[1]*4,p)
    elif spatial_rank in [3]:
      conv_func = tf.nn.conv3d
      if upscale=='conv_transpose':
        conv_t_func = tf.nn.conv3d_transpose
      else:
        def resize3(x,sz):
          r1=tf.image.resize_images(tf.reshape(x,x.shape.as_list()[:3]+[-1]),sz[1:3])
          r2=tf.image.resize_images(tf.reshape(r1,[sz[0],sz[1]*sz[2],x.shape.as_list()[3],-1]),[sz[1]*sz[2],sz[3]])
          return tf.reshape(r2,sz[:4]+[x.shape.as_list()[-1]])
        conv_t_func=lambda x, Wt, sz, st, p: conv_func(resize3(x,sz),tf.transpose(Wt,[0,1,2,4,3]),[1]*5,p)
    conv_t_unit = lambda x,Wt,sz,norm_func: acti_func(norm_func(conv_t_func(x, Wt, [batch_size]+sz, [1]+[2]*spatial_rank+[1], "SAME")))
    conv_unit = lambda x,W,sz,norm_func: acti_func(norm_func(conv_func(x, W, [1]+[1]*spatial_rank+[1], "SAME")))
    upsample_unit = lambda x,Wt,W,sz: conv_unit(conv_t_unit(x,Wt,sz,norm_func),W,sz,norm_func)
    last_upsample_unit = lambda x,Wt,W,sz: conv_func(conv_t_unit(x,Wt,sz,norm_func),W,[1]+[1]*spatial_rank+[1], "SAME")
    kernel_size=[3]*spatial_rank

    for it in range(self._num_layers):
      Wt=tf.get_variable('G_Wt{}'.format(it),shape=kernel_size+[intermediate_sizes[it+1][-1],intermediate_sizes[it][-1]],initializer=self.initializers['w'])
      W =tf.get_variable( 'G_W{}'.format(it),shape=kernel_size+[intermediate_sizes[it+1][-1],intermediate_sizes[it+1][-1]],initializer=self.initializers['w'])
      if it<self._num_layers-1:
        image=upsample_unit(image,Wt,W,intermediate_sizes[it+1])
      else:
        image=last_upsample_unit(image,Wt,W,intermediate_sizes[it+1]) # NB. no batch_norm for true mean and scale
    channel_scale = tf.get_variable('G_scale',shape=[1]*(spatial_rank+1)+[intermediate_sizes[-1][-1]],initializer=tf.constant_initializer(.1))
    #channel_shift = tf.get_variable('G_shift',shape=[1]*(spatial_rank+1)+[intermediate_sizes[-1][-1]],initializer=tf.constant_initializer(0.))
    channel_shift = tf.get_variable('G_shift',shape=[1]+intermediate_sizes[-1],initializer=tf.constant_initializer(0.))

    image = image*channel_scale+channel_shift
    return tf.nn.tanh(image)

class ImageDiscriminator(TrainableLayer):
  """
    ### Description

    ### Diagram

    ### Constraints

  """
  def __init__(self, hidden_layer_channels,name):
    """

    :param hidden_layer_channels: array, number of output channels for each layer
    :param name: layer name
    """
    super(ImageDiscriminator, self).__init__(name=name)
    self._layer_channels = hidden_layer_channels
    self.initializers = {'w':tf.contrib.layers.variance_scaling_initializer(),'b':tf.constant_initializer(0)}

  def layer_op(self, image,is_training):
    """

    :param image: tensor, input image to distriminator
    :param is_training: boolean, True if network is in training mode
    :return: tensor, classification logits
    """
    batch_size=image.shape.as_list()[0]
    spatial_rank=len(image.shape)-2
    image_channels = image.shape.as_list()[-1]
    acti_func=tf.nn.relu
    dropout_func = lambda x: tf.nn.dropout(x,.5)
    norm_func = tf.contrib.layers.batch_norm
    downscale=['resize','conv_stride'][0]
    if spatial_rank==2:
      conv_func=tf.nn.conv2d
      if downscale=='conv_stride':
        conv_s_unit = lambda x,W: acti_func(norm_func(conv_func(x, W, [1]+[2]*spatial_rank+[1], "SAME")))
      else:
        conv_s_unit = lambda x,W: acti_func(norm_func(conv_func(tf.image.resize_images(x,[x.shape.as_list()[1]//2,x.shape.as_list()[2]//2]), W, [1]+[1]*spatial_rank+[1], "SAME")))
    elif spatial_rank in [3]:
      conv_func = tf.nn.conv3d
      if downscale=='conv_stride':
        conv_s_unit = lambda x,W: acti_func(norm_func(conv_func(x, W, [1]+[2]*spatial_rank+[1], "SAME")))
      else:
        def resize3(x,sz):
          r1=tf.image.resize_images(tf.reshape(x,x.shape.as_list()[:3]+[-1]),sz[1:3])
          r2=tf.image.resize_images(tf.reshape(r1,[sz[0],sz[1]*sz[2],x.shape.as_list()[3],-1]),[sz[1]*sz[2],sz[3]])
          return tf.reshape(r2,sz[:4]+[x.shape.as_list()[-1]])
        conv_s_unit = lambda x,W: acti_func(norm_func(conv_func(resize3(x,[a//b for a,b in zip(x.shape.as_list(),[1,2,2,2,1])]), W, [1]+[1]*spatial_rank+[1], "SAME")))
    conv_unit = lambda x,W: acti_func(norm_func(conv_func(x, W, [1]+[1]*spatial_rank+[1], "SAME")))
    down_sample_unit = lambda x,Ws,W: conv_s_unit(conv_unit(x,W),Ws)
    kernel_size=[3]*spatial_rank
    for it in range(len(self._layer_channels)-1):
      if it==0:
        W =tf.get_variable( 'D_W{}'.format(it),shape=kernel_size+[image_channels,self._layer_channels[it]],initializer=self.initializers['w'])
      else:
        W =tf.get_variable( 'D_W{}'.format(it),shape=kernel_size+[self._layer_channels[it],self._layer_channels[it]],initializer=self.initializers['w'])
      Ws=tf.get_variable('D_Ws{}'.format(it),shape=kernel_size+[self._layer_channels[it],self._layer_channels[it+1]],initializer=self.initializers['w'])
      image=down_sample_unit(image,Ws,W)
    logits = tf.layers.dense(tf.reshape(image,[batch_size,-1]),1,activation=None,use_bias=True)
    return logits

