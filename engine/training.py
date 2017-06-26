# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import time

import numpy as np
import tensorflow as tf
from six.moves import range

from engine.input_buffer import TrainEvalInputBuffer
from engine.spatial_location_check import SpatialLocationCheckLayer
from engine.selective_sampler import SelectiveSampler
from engine.uniform_sampler import UniformSampler
from layer.loss import LossFunction
from utilities import misc_common as util
from utilities.input_placeholders import ImagePatch
from utilities.training_output import QuantitiesToMonitor

np.random.seed(seed=int(time.time()))


def run(net_class, param, volume_loader, device_str):
    # construct graph
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        # defines a training element
        patch_holder = ImagePatch(
            spatial_rank=param.spatial_rank,
            image_size=param.image_size,
            label_size=param.label_size,
            weight_map_size=param.w_map_size,
            image_dtype=tf.float32,
            label_dtype=tf.int64,
            weight_map_dtype=tf.float32,
            num_image_modality=volume_loader.num_modality(0),
            num_label_modality=volume_loader.num_modality(1),
            num_weight_map=volume_loader.num_modality(2))
        # defines data augmentation for training
        augmentations = []
        if param.rotation:
            from layer.rand_rotation import RandomRotationLayer
            augmentations.append(RandomRotationLayer(
                min_angle=param.min_angle,
                max_angle=param.max_angle))
        if param.spatial_scaling:
            from layer.rand_spatial_scaling import RandomSpatialScalingLayer
            augmentations.append(RandomSpatialScalingLayer(
                min_percentage=param.min_percentage,
                max_percentage=param.max_percentage))
        # defines how to generate samples of the training element from volume
        if param.window_sampling == 'uniform':
            sampler = UniformSampler(patch=patch_holder,
                                     volume_loader=volume_loader,
                                     patch_per_volume=param.sample_per_volume,
                                     data_augmentation_methods=augmentations,
                                     name='uniform_sampler')
        elif param.window_sampling == 'selective':
            # TODO check param, this is for segmentation problems only
            spatial_location_check = SpatialLocationCheckLayer(
                compulsory=((0), (0)),
                minimum_ratio=param.min_sampling_ratio,
                min_numb_labels=param.min_numb_labels,
                padding=param.border,
                name='spatial_location_check')
            sampler = SelectiveSampler(
                patch=patch_holder,
                volume_loader=volume_loader,
                spatial_location_check=spatial_location_check,
                data_augmentation_methods=None,
                patch_per_volume=param.sample_per_volume,
                name="selective_sampler")

        w_regularizer = None
        b_regularizer = None
        if param.reg_type.lower() == 'l2':
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l2_regularizer(param.decay)
            b_regularizer = regularizers.l2_regularizer(param.decay)
        elif param.reg_type.lower() == 'l1':
            from tensorflow.contrib.layers.python.layers import regularizers
            w_regularizer = regularizers.l1_regularizer(param.decay)
            b_regularizer = regularizers.l1_regularizer(param.decay)
        net = net_class(num_classes=param.num_classes,
                        w_regularizer=w_regularizer,
                        b_regularizer=b_regularizer,
                        acti_func=param.activation_function)
        loss_func = LossFunction(n_class=param.num_classes,
                                 loss_type=param.loss_type)
        # construct train queue
        train_batch_runner = TrainEvalInputBuffer(
            batch_size=param.batch_size,
            capacity=max(param.queue_length, param.batch_size),
            sampler=sampler,
            shuffle=True)
        # optimizer
        train_step = tf.train.AdamOptimizer(learning_rate=param.lr)
        tower_losses, tower_grads = [], []
        train_pairs = train_batch_runner.pop_batch_op
        images, labels = train_pairs['images'], train_pairs['labels']
        if "weight_maps" in train_pairs:
            weight_maps = train_pairs['weight_maps']
        else:
            weight_maps = None
        for i in range(0, max(param.num_gpus, 1)):
            with tf.device("/{}:{}".format(device_str, i)):
                predictions = net(images, is_training=True)
                loss = loss_func(predictions, labels, weight_maps)
                if param.decay > 0:
                    reg_losses = graph.get_collection(
                        tf.GraphKeys.REGULARIZATION_LOSSES)
                    reg_loss = tf.reduce_mean([tf.reduce_mean(reg_loss)
                                               for reg_loss in reg_losses])
                    loss = loss + reg_loss
                # TODO compute miss for dfferent target types
                grads = train_step.compute_gradients(loss)
                tower_losses.append(loss)
                # Get the list of quantities to monitor during training
                if hasattr(net, 'quantities_to_monitor'):
                    [tower_additional, tower_additional_names] = QuantitiesToMonitor(predictions, labels,
                                                                                     net.quantities_to_monitor)
                else:
                    [tower_additional, tower_additional_names] = QuantitiesToMonitor(predictions, labels, {})
                tower_grads.append(grads)
                # note: only use batch stats from one GPU for batch_norm
                if i == 0:
                    bn_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        ave_loss = tf.reduce_mean(tower_losses)
        average_additional = []
        for p in range(0, len(tower_additional)):
            average_additional.append(tf.reduce_mean(tower_additional[p]))
        ave_grads = util.average_grads(tower_grads)
        apply_grad_op = train_step.apply_gradients(ave_grads)
        # summary for visualisations
        # tracking current batch loss
        summaries = [tf.summary.scalar("total-loss", ave_loss)]
        for p in range(0, len(tower_additional)):
            summaries += [tf.summary.scalar(tower_additional_names[p], tower_additional[p])]
        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9)
        var_averages_op = variable_averages.apply(tf.trainable_variables())
        # batch norm variables moving mean and var
        batchnorm_updates_op = tf.group(*bn_updates)
        # primary operations
        init_op = tf.global_variables_initializer()
        train_op = tf.group(apply_grad_op,
                            var_averages_op,
                            batchnorm_updates_op)
        write_summary_op = tf.summary.merge(summaries)
        # saver
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(max_to_keep=20, var_list=variables_to_restore)
        tf.Graph.finalize(graph)
    # run session
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True
    start_time = time.time()
    with tf.Session(config=config, graph=graph) as sess:
        # prepare output directory
        if not os.path.exists(os.path.join(param.model_dir, 'models')):
            os.makedirs(os.path.join(param.model_dir, 'models'))
        root_dir = os.path.abspath(param.model_dir)
        # start or load session
        ckpt_name = os.path.join(root_dir, 'models', 'model.ckpt')
        if param.starting_iter > 0:
            model_str = '{}-{}'.format(ckpt_name, param.starting_iter)
            saver.restore(sess, model_str)
            print('Loading from {}...'.format(model_str))
        else:
            sess.run(init_op)
            print('Weights from random initialisations...')
        coord = tf.train.Coordinator()
        writer = tf.summary.FileWriter(os.path.join(root_dir, 'logs'),
                                       sess.graph)
        try:
            print('Filling the queue (this can take a few minutes)')
            train_batch_runner.run_threads(sess, coord, param.num_threads)
            for i in range(param.max_iter - param.starting_iter):
                local_time = time.time()
                if coord.should_stop():
                    break
                values = sess.run([train_op, ave_loss] + average_additional)
                current_iter = i + param.starting_iter
                iter_time = time.time() - local_time
                output_string = 'iter {:d}, loss={:.8f}'
                for p in range(0, len(tower_additional)):
                    output_string += ', ' + tower_additional_names[p] + '={:.8f}'
                output_string += ' ({:.3f}s)'
                format_string = [current_iter] + values[1::] + [iter_time]
                print(output_string.format(*format_string))
                if (current_iter % 20) == 0:
                    writer.add_summary(sess.run(write_summary_op), current_iter)
                if (current_iter % param.save_every_n) == 0 and i > 0:
                    saver.save(sess, ckpt_name, global_step=current_iter)
                    print('Iter {} model saved at {}'.format(
                        current_iter, ckpt_name))
        except KeyboardInterrupt:
            print('User cancelled training')
        except tf.errors.OutOfRangeError as e:
            pass
        except Exception:
            import sys, traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=sys.stdout)
        finally:
            saver.save(sess, ckpt_name, global_step=param.max_iter)
            print('Last iteration model saved at {}'.format(ckpt_name))
            print('training.py (time in second) {:.2f}'.format(
                time.time() - start_time))
            train_batch_runner.close_all()
