# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import time

import numpy as np
import tensorflow as tf
from six.moves import range

from niftynet.engine.input_buffer import TrainEvalInputBuffer
from niftynet.engine.spatial_location_check import SpatialLocationCheckLayer
from niftynet.engine.selective_sampler import SelectiveSampler
from niftynet.engine.uniform_sampler import UniformSampler
from niftynet.engine.resize_sampler import ResizeSampler
from niftynet.layer.loss import LossFunction
from niftynet.utilities import misc_common as util
from niftynet.utilities.input_placeholders import ImagePatch
import niftynet.engine.logging
from niftynet.engine.restorer import global_variables_initialize_or_restorer

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
        if param.random_flip:
            from niftynet.layer.rand_flip import RandomFlipLayer
            augmentations.append(RandomFlipLayer(
                flip_axes=param.flip_axes))
        if param.rotation:
            from niftynet.layer.rand_rotation import RandomRotationLayer
            augmentations.append(RandomRotationLayer(
                min_angle=param.min_angle,
                max_angle=param.max_angle))
        if param.spatial_scaling:
            from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
            augmentations.append(RandomSpatialScalingLayer(
                min_percentage=param.min_percentage,
                max_percentage=param.max_percentage))
        # defines how to generate samples of the training element from volume
        with tf.name_scope('Sampling'):
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
                    data_augmentation_methods=augmentations,
                    patch_per_volume=param.sample_per_volume,
                    name="selective_sampler")
            elif param.window_sampling == 'resize':
                sampler = ResizeSampler(
                    patch=patch_holder,
                    volume_loader=volume_loader,
                    data_augmentation_methods=augmentations,
                    name="resize_sampler")
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
        with tf.name_scope('DataQueue'):
            train_batch_runner = TrainEvalInputBuffer(
                batch_size=param.batch_size,
                capacity=max(param.queue_length, param.batch_size),
                sampler=sampler,
                shuffle=True)
        # optimizer
        with tf.name_scope('Optimizer'):
            train_step = tf.train.AdamOptimizer(learning_rate=param.lr)
        tower_grads = []
        # Scalar summaries for the console are averaged over GPU runs
        console_outputs=graph.get_collection_ref(niftynet.engine.logging.CONSOLE)
        console_outputs_cache=console_outputs[:]
        del console_outputs[:]
        tower_console_outputs=[]

        for i in range(0, max(param.num_gpus, 1)):
            with tf.device("/{}:{}".format(device_str, i)):

                train_pairs = train_batch_runner.pop_batch_op(i)
                images, labels = train_pairs['Sampling/images'], train_pairs['Sampling/labels']
                if "weight_maps" in train_pairs:
                    weight_maps = train_pairs['Sampling/weight_maps']
                else:
                    weight_maps = None

                predictions = net(images, is_training=True)
                with tf.name_scope('Loss'):
                    loss = loss_func(predictions, labels, weight_maps)
                    if param.decay > 0:
                        reg_losses = graph.get_collection(
                            tf.GraphKeys.REGULARIZATION_LOSSES)
                        if reg_losses:
                            reg_loss = tf.reduce_mean([tf.reduce_mean(reg_loss)
                                                    for reg_loss in reg_losses])
                            loss = loss + reg_loss

                ##################
                # This should probably be refactored into an application class
                # Averages are in name_scope for Tensorboard naming; summaries are outside for console naming
                with tf.name_scope('ConsoleLogging'):
                    logs=[['loss',loss]]
                    if param.application_type == 'segmentation':
                        # TODO compute miss for dfferent target types
                        logs.append(['miss', tf.reduce_mean(tf.cast(
                              tf.not_equal(tf.argmax(predictions, -1), labels[..., 0]),
                              dtype=tf.float32))])
                for tag,val in logs:
                    tf.summary.scalar(tag,val,[niftynet.engine.logging.CONSOLE,niftynet.engine.logging.LOG])
                ##################

                # record and clear summaries
                console_outputs=graph.get_collection_ref(niftynet.engine.logging.CONSOLE)
                tower_console_outputs.append(console_outputs[:])
                del console_outputs[:]

                with tf.name_scope('ComputeGradients'):
                    grads = train_step.compute_gradients(loss)
                tower_grads.append(grads)
                # note: only use batch stats from one GPU for batch_norm
                if i == 0:
                    bn_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.name_scope('AccumulateGradients'):
            ave_grads = util.average_grads(tower_grads)
            apply_grad_op = train_step.apply_gradients(ave_grads)

        # Add averaged summaries
        console_outputs=graph.get_collection_ref(niftynet.engine.logging.CONSOLE)
        console_outputs+=console_outputs_cache
        if len(tower_console_outputs)>1:
            # Averages are in name_scope for Tensorboard naming; summaries are outside for console naming
            with tf.name_scope('AccumulateConsoleLogs'):
                averaged_summaries=[]
                for replicated_output in zip(*tower_console_outputs):
                    averaged_summaries.append([replicated_output[0].op.name+'_avg',tf.reduce_mean([o.op.inputs[1] for o in replicated_output])])
            for tag,avg in averaged_summaries:
                tf.summary.scalar(tag, avg,[niftynet.engine.logging.CONSOLE,niftynet.engine.logging.LOG])
        else:
            console_outputs+=tower_console_outputs[0]
        # Track the moving averages of all trainable variables.
        with tf.name_scope('MovingAverages'):
            variable_averages = tf.train.ExponentialMovingAverage(0.9)
            var_averages_op = variable_averages.apply(tf.trainable_variables())
            # batch norm variables moving mean and var
            batchnorm_updates_op = tf.group(*bn_updates)
        # primary operations
        init_op = global_variables_initialize_or_restorer()
        train_op = tf.group(apply_grad_op,
                            var_averages_op,
                            batchnorm_updates_op)
        logged_summaries = list(set([s for c in [niftynet.engine.logging.LOG,niftynet.engine.logging.CONSOLE] for s in tf.get_collection(c)]))
        write_summary_op = tf.summary.merge(logged_summaries)
        # saver
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(max_to_keep=param.max_checkpoints)
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
        # Place logs from each new training run in a new folder for better tensorboard visualization
        if not os.path.exists(os.path.join(root_dir, 'logs')):
          os.makedirs(os.path.join(root_dir, 'logs'))
        log_sub_dirs = [name for name in os.listdir(os.path.join(root_dir, 'logs')) if name.isdecimal()]
        if log_sub_dirs and param.starting_iter==0:
            log_sub_dir = str(max([int(name) for name in log_sub_dirs])+1)
        elif log_sub_dirs and param.starting_iter > 0:
            log_sub_dir = str(max([int(name) for name in log_sub_dirs if os.path.isdir(os.path.join(root_dir, 'logs', name))]))
        else:
            log_sub_dir = '0'
        writer = tf.summary.FileWriter(os.path.join(root_dir, 'logs', log_sub_dir),
                                       sess.graph)
        try:
            print('Filling the queue (this can take a few minutes)')
            train_batch_runner.run_threads(sess, coord, param.num_threads)
            for i in range(param.max_iter - param.starting_iter):
                local_time = time.time()
                if coord.should_stop():
                    break
                current_iter = i + param.starting_iter
                ops_to_run=[train_op]
                console_summaries=tf.get_collection(niftynet.engine.logging.CONSOLE)
                ops_to_run += console_summaries
                if (current_iter % 20) == 0:
                    ops_to_run += [write_summary_op]
                values = sess.run(ops_to_run)[1:]
                if (current_iter % 20) == 0:
                    writer.add_summary(values.pop(), current_iter)
                summary_string = ''.join([niftynet.engine.logging.console_summary_string(v) for v in values])
                iter_time = time.time() - local_time
                print(('iter {:d}{}, ({:.3f}s)').format(
                    current_iter, summary_string, iter_time))
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
