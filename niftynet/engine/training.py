# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import time
import itertools

import numpy as np
import tensorflow as tf
from six.moves import range

from niftynet.engine.input_buffer import TrainEvalInputBuffer
from niftynet.utilities import misc_common as util
from niftynet.utilities.input_placeholders import ImagePatch
import niftynet.engine.logging
from niftynet.application.common import ApplicationFactory
        
np.random.seed(seed=int(time.time()))


def run(net_class, param, volume_loader, device_str):
    application=ApplicationFactory(param)(net_class, param, volume_loader)
    # construct graph
    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        # defines a training element
        sampler = application.sampler()
        # construct train queue
        with tf.name_scope('DataQueue'):
            train_batch_runner = TrainEvalInputBuffer(
                batch_size=param.batch_size,
                capacity=max(param.queue_length, param.batch_size),
                sampler=sampler,
                shuffle=True)
        tower_grads = []
        train_dict = train_batch_runner.pop_batch_op
        # Scalar summaries for the console are averaged over GPU runs
        console_outputs=graph.get_collection_ref(niftynet.engine.logging.CONSOLE)
        console_outputs_cache=console_outputs[:]
        del console_outputs[:]
        tower_console_outputs=[]
        
        for i in range(0, max(param.num_gpus, 1)):
            with tf.device("/{}:{}".format(device_str, i)):
                grads = application.train(train_dict)
                tower_grads.append(grads)
                # note: only use batch stats from one GPU for batch_norm
                if i == 0:
                    bn_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                # record and clear summaries
                console_outputs=graph.get_collection_ref(niftynet.engine.logging.CONSOLE)
                tower_console_outputs.append(console_outputs[:])
                del console_outputs[:]
        # group gradient ops by op_type, not GPU
        tower_grads=list(zip(*tower_grads))
        # add apply_grad_op for each type of optimizer_op
        apply_grad_ops=[]
        with tf.name_scope('AccumulateGradients'):
          for idx,ops in enumerate(tower_grads):
            ave_grads = util.average_grads(ops)
            apply_grad_ops.append(application.optimizer.apply_gradients(ave_grads))

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
        init_op = tf.global_variables_initializer()
        train_ops = [tf.group(apply_grad_op,
                            var_averages_op,
                            batchnorm_updates_op) for apply_grad_op in apply_grad_ops]
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
        writer = tf.summary.FileWriter(os.path.join(root_dir, 'logs'),
                                       sess.graph)
        try:
            print('Filling the queue (this can take a few minutes)')
            train_batch_runner.run_threads(sess, coord, param.num_threads)
            # Flush the train_ops up to training iter (for applications with complex training protocols)
            train_op_sequence = application.train_op_generator(train_ops)
            _=itertools.islice(train_op_sequence,param.starting_iter)
            for i in range(param.max_iter - param.starting_iter):
                local_time = time.time()
                if coord.should_stop():
                    break
                current_iter = i + param.starting_iter
                train_op_list = train_op_sequence.__next__()
                for idx,train_op in enumerate(train_op_list):
                    ops_to_run=[train_op]
                    if idx==0:
                        console_summaries=tf.get_collection(niftynet.engine.logging.CONSOLE)
                        ops_to_run += console_summaries
                        if (current_iter % 20) == 0:
                            ops_to_run += [write_summary_op]
                    values = sess.run(ops_to_run)[1:]
                    if idx==0 and (current_iter % 20) == 0:
                        writer.add_summary(values.pop(), current_iter)
                    summary_string = ''.join([niftynet.engine.logging.console_summary_string(v) for v in values])
                    iter_time = time.time() - local_time
                    iter_str = '{:d}'.format(current_iter) if len(train_op_list)==1 else '{:d}.{:d}'.format(current_iter, idx)
                    print(('iter {}{}, ({:.3f}s)').format(
                        iter_str, summary_string, iter_time))
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
