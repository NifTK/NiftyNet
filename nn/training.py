# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import range

from network.net_template import NetTemplate as NetTemplate
from utilities import misc as util
from .input_queue import TrainEvalInputBuffer
from .loss import LossFunction
from .sampler import VolumeSampler

np.random.seed(seed=int(time.time()))


def run(net, param):
    if not isinstance(net, NetTemplate):
        print('net model should inherit network.NetTemplate')
        return
    assert (param.batch_size <= param.queue_length)
    patient_list = util.list_patient(param.train_data_dir)
    modality_list = util.list_modality(param.train_data_dir)
    rand_sampler = VolumeSampler(
        patient_list,
        modality_list,
        net.batch_size,
        net.input_image_size,
        net.input_label_size,
        param.volume_padding_size,
        param.histogram_ref_file,
        param.sample_per_volume)
    sample_generator = rand_sampler.uniform_sampling_from(param.train_data_dir)

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        # construct train queue and graph
        input_shapes = [
            # multimodal image size
            [param.image_size] * 3 + [len(rand_sampler.modalities)],
            # label size
            [param.image_size] * 3,
            # location info size
            [7]]
        train_batch_runner = TrainEvalInputBuffer(
            batch_size=param.batch_size,
            capacity=param.queue_length,
            shapes=input_shapes,
            sample_generator=sample_generator,
            shuffle=True)
        loss_func = LossFunction(net.num_classes,
                                 param.loss_type,
                                 param.reg_type,
                                 param.decay)
        # optimizer
        train_step = tf.train.AdamOptimizer(learning_rate=param.lr)

        tower_misses = []
        tower_losses = []
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(param.num_gpus):
                train_pairs = train_batch_runner.pop_batch_op
                images = train_pairs['images']
                labels = train_pairs['labels']
                with tf.device("/gpu:%d" % i), tf.name_scope("N_%d" % i) as scope:
                    predictions = net.inference(images)
                    loss = loss_func.total_loss(predictions, labels, scope)
                    miss = tf.reduce_mean(tf.cast(
                        tf.not_equal(tf.argmax(predictions, -1), labels),
                        dtype=tf.float32))

                    tf.get_variable_scope().reuse_variables()
                    grads = train_step.compute_gradients(loss)
                    tower_losses.append(loss)
                    tower_misses.append(miss)
                    tower_grads.append(grads)
        ave_loss = tf.reduce_mean(tower_losses)
        ave_miss = tf.reduce_mean(tower_misses)
        ave_grads = util.average_grads(tower_grads)
        apply_grad_op = train_step.apply_gradients(ave_grads)

        # summary for visualisations
        # tracking current batch loss
        summaries = []
        summaries.append(tf.summary.scalar("total-miss", ave_miss))
        summaries.append(tf.summary.scalar("total-loss", ave_loss))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(0.9)
        var_averages_op = variable_averages.apply(tf.trainable_variables())

        # primary operations
        init_op = tf.global_variables_initializer()
        train_op = tf.group(apply_grad_op, var_averages_op)
        write_summary_op = tf.summary.merge(summaries)
        # saver
        saver = tf.train.Saver(max_to_keep=20)
        tf.Graph.finalize(graph)

    # run session
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True

    start_time = time.time()
    with tf.Session(config=config, graph=graph) as sess:
        # prepare output directory
        if not os.path.exists(param.model_dir + '/models'):
            os.makedirs(param.model_dir + '/models')
        root_dir = os.path.abspath(param.model_dir)
        # start or load session
        ckpt_name = root_dir + '/models/model.ckpt'
        if param.starting_iter > 0:
            model_str = ckpt_name + '-%d' % (param.starting_iter)
            saver.restore(sess, model_str)
            print('Loading from {}...'.format(model_str))
        else:
            sess.run(init_op)
            print('Weights from random initialisations...')

        coord = tf.train.Coordinator()
        writer = tf.summary.FileWriter(root_dir + '/logs', sess.graph)
        try:
            print('Filling the queue (this can take a few minutes)')
            train_batch_runner.run_threads(sess, coord, param.num_threads)
            for i in range(param.max_iter):
                local_time = time.time()
                if coord.should_stop():
                    break
                _, loss_value, miss_value = sess.run([train_op,
                                                      ave_loss,
                                                      ave_miss])
                print('iter {:d}, loss={:.8f},' \
                      'error_rate={:.8f} ({:.3f}s)'.format(
                    i, loss_value, miss_value, time.time() - local_time))
                if (i % 20) == 0:
                    writer.add_summary(sess.run(write_summary_op),
                                       i + param.starting_iter)
                if (i % param.save_every_n) == 0 and (i > 0):
                    saver.save(sess, ckpt_name,
                               global_step=i + param.starting_iter)
                    print('Iter {} model saved at {}'.format(
                        i + param.starting_iter, ckpt_name))
        except KeyboardInterrupt:
            print('User cancelled training')
        except tf.errors.OutOfRangeError:
            pass
        except ValueError as e:
            print(e)
        except RuntimeError as e:
            print(e)
        finally:
            saver.save(sess, ckpt_name,
                       global_step=param.max_iter + param.starting_iter)
            print('Last iteration model saved at {}'.format(ckpt_name))
            print('training.py (time in second) {:.2f}'.format(
                time.time() - start_time))
            train_batch_runner.close_all()
