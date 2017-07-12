from niftynet.engine.spatial_location_check import SpatialLocationCheckLayer
from niftynet.engine.gan_sampler import GANSampler
from niftynet.layer.gan_loss import LossFunction
from niftynet.utilities.input_placeholders import GANPatch
import numpy as np
import scipy
import tensorflow as tf
from niftynet.engine import logging
from niftynet.layer.post_processing import PostProcessingLayer
from niftynet.application.common import BaseApplication

import time
class GANApplication(BaseApplication):
  def __init__(self,net_class, param, volume_loader):
    self._net_class = net_class
    self._param = param
    self._volume_loader = volume_loader
    self._loss_func = LossFunction(loss_type=self._param.loss_type)
    self.num_objectives=2
    w_regularizer,b_regularizer=self.regularizers()
    self._net=None
  def inference_sampler(self):
        self._inference_patch_holder = GANPatch(
            spatial_rank=self._param.spatial_rank,
            image_size=self._param.image_size,
            noise_size=self._param.noise_size,
            conditioning_size=self._param.conditioning_size,
            num_image_modality=self._volume_loader.num_modality(0))

        sampler = GANSampler(
                patch=self._inference_patch_holder,
                volume_loader=self._volume_loader,
                data_augmentation_methods=None)
            # ops to resize image back
        self._ph=tf.placeholder(tf.float32,[None])
        self._sz=tf.placeholder(tf.int32,[None])
        reshaped=tf.image.resize_images(tf.reshape(self._ph,[1]+[self._param.label_size]*2+[-1]),self._sz[0:2])
        if self._param.spatial_rank==3:
            reshaped=tf.reshape(reshaped,[1,self._sz[0]*self._sz[1],self._param.label_size,-1])
            reshaped=tf.image.resize_images(reshaped,[self._sz[0]*self._sz[1],self._sz[2]])
        self._reshaped=tf.reshape(reshaped,self._sz)
        return sampler
    
  def sampler(self):
        patch_holder = GANPatch(
            spatial_rank=self._param.spatial_rank,
            image_size=self._param.image_size,
            noise_size=self._param.noise_size,
            conditioning_size=self._param.conditioning_size,
            num_image_modality=self._volume_loader.num_modality(0))
        # defines data augmentation for training
        augmentations = []
        if self._param.rotation:
            from niftynet.layer.rand_rotation import RandomRotationLayer
            augmentations.append(RandomRotationLayer(
                min_angle=self._param.min_angle,
                max_angle=self._param.max_angle))
        if self._param.spatial_scaling:
            from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
            augmentations.append(RandomSpatialScalingLayer(
                min_percentage=self._param.min_percentage,
                max_percentage=self._param.max_percentage))
        # defines how to generate samples of the training element from volume
        with tf.name_scope('Sampling'):
            sampler = GANSampler(
                patch=patch_holder,
                volume_loader=self._volume_loader,
                data_augmentation_methods=None)
        return sampler
  def net(self,train_dict,is_training):
    if not self._net:
      self._net=self._net_class()
    return self._net(train_dict['Sampling/noise'],train_dict['Sampling/images'],is_training)
  def net_inference(self,train_dict,is_training):
    if not self._net:
      self._net=self._net_class()
    net_outputs = self._net(train_dict['images'],is_training)
    return self._post_process_outputs(net_outputs),train_dict['info']
  def loss_func(self,train_dict,net_outputs):
    real_logits = net_outputs[1]
    fake_logits = net_outputs[2]
    diff = net_outputs[3]
    lossG = self._loss_func(fake_logits,True)
    lossD=self._loss_func(real_logits,True)+self._loss_func(fake_logits,False)
    lossL2 = tf.reduce_mean(tf.square(diff))
    return lossG,lossD,lossL2
  def train(self,train_dict):
    """
    Returns a list of possible compute_gradients ops to be run each training iteration.
    Default implementation returns gradients for all variables from one Adam optimizer
    """
    # optimizer
    with tf.name_scope('Optimizer'):
      self.optimizer = tf.train.AdamOptimizer(learning_rate=self._param.lr,)
    net_outputs = self.net(train_dict, is_training=True)
    with tf.name_scope('Loss'):
        lossG,lossD,lossL2 = self.loss_func(train_dict,net_outputs)
        if self._param.decay > 0:
            reg_losses = tf.get_collection(
                tf.GraphKeys.REGULARIZATION_LOSSES)
            if reg_losses:
                reg_loss = tf.reduce_mean([tf.reduce_mean(reg_loss)
                                    for reg_loss in reg_losses])
                lossD = lossD + reg_loss
                lossG = lossG + reg_loss
    # Averages are in name_scope for Tensorboard naming; summaries are outside for console naming
    logs=[['lossD',lossD],['lossG',lossG]]
    with tf.name_scope('ConsoleLogging'):
        logs+=self.logs(train_dict,net_outputs)
    for tag,val in logs:
        tf.summary.scalar(tag,val,[logging.CONSOLE,logging.LOG])
    with tf.name_scope('ComputeGradients'):
      grads=[self.optimizer.compute_gradients(lossG,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator')),
             self.optimizer.compute_gradients(lossD,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='discriminator')),
             self.optimizer.compute_gradients(lossL2,var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='generator/G_shift'))]   
    # add compute gradients ops for each type of optimizer_op
    return grads

  def inference_loop(self, sess, coord, net_out):
    all_saved_flag = False
    img_id, pred_img, subject_i = None, None, None
    spatial_rank = self._inference_patch_holder.spatial_rank
    while True:
        local_time = time.time()
        if coord.should_stop():
            break
        seg_maps, spatial_info = sess.run(net_out)
        # go through each one in a batch
        for batch_id in range(seg_maps.shape[0]):
            img_id = spatial_info[batch_id, 0]
            subject_i = self._volume_loader.get_subject(img_id)
            pred_img = subject_i.matrix_like_input_data_5d(
                    spatial_rank=spatial_rank,
                    n_channels=self._num_output_channels_func(),
                    interp_order=self._param.output_interp_order)
            predictions = seg_maps[batch_id]
            while predictions.ndim < pred_img.ndim:
                predictions = np.expand_dims(predictions, axis=-1)

            # assign predicted patch to the allocated output volume
            origin = spatial_info[
                     batch_id, 1:(1 + int(np.floor(spatial_rank)))]

            i_spatial_rank=int(np.ceil(spatial_rank))
            zoom=[d/p for p,d in zip([self._param.label_size]*i_spatial_rank,pred_img.shape[0:i_spatial_rank])]+[1,1]
            pred_img=sess.run([self._reshaped],feed_dict={self._ph:np.reshape(predictions,[-1]),self._sz:pred_img.shape})[0]
            subject_i.save_network_output(
                        pred_img,
                        self._param.save_seg_dir,
                        self._param.output_interp_order)

            if self._inference_patch_holder.is_stopping_signal(
                        spatial_info[batch_id]):
                    print('received finishing batch')
                    all_saved_flag = True
                    return all_saved_flag

            # try to expand prediction dims to match the output volume
        print('processed {} image patches ({:.3f}s)'.format(
            len(spatial_info), time.time() - local_time))  
    return all_saved_flag
  def logs(self,train_dict,net_outputs):
    return []
  def train_op_generator(self,apply_ops):
    for it in range(100):
      yield apply_ops[2:]
    for it in range(100):
      yield apply_ops[1:2]
    while True:
      yield apply_ops[:2]
