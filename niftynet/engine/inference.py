# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import os.path
import time

import numpy as np
import tensorflow as tf
from six.moves import range

from niftynet.engine.input_buffer import DeployInputBuffer
from niftynet.utilities.input_placeholders import ImagePatch
from niftynet.application.common import ApplicationFactory

# run on single GPU with single thread
def run(net_class, param, volume_loader, device_str):
    application=ApplicationFactory(param)(net_class, param, volume_loader)
    # construct graph
    graph = tf.Graph()
    with graph.as_default(), tf.device("/{}:0".format(device_str)):
        # construct inference queue and graph
        # TODO change batch size param - batch size could be larger in test case
        sampler = application.inference_sampler()

        # construct train queue
        seg_batch_runner = DeployInputBuffer(
            batch_size=param.batch_size,
            capacity=max(param.queue_length, param.batch_size),
            sampler=sampler)
        inference_dict = seg_batch_runner.pop_batch_op()
        net_out = application.net_inference(inference_dict, is_training=False)
        variable_averages = tf.train.ExponentialMovingAverage(0.9)
        saver = tf.train.Saver()
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(var_list=variables_to_restore)
        tf.Graph.finalize(graph)  # no more graph nodes after this line

    # run session
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.allow_soft_placement = True
    # config.gpu_options.allow_growth = True

    start_time = time.time()
    with tf.Session(config=config, graph=graph) as sess:
        root_dir = os.path.abspath(param.model_dir)
        print('Model folder {}'.format(root_dir))
        if not os.path.exists(root_dir):
            raise ValueError('Model folder not found {}'.format(root_dir))
        ckpt = tf.train.get_checkpoint_state(os.path.join(root_dir, 'models'))
        if ckpt and ckpt.model_checkpoint_path:
            print('Evaluation from checkpoints')
        model_str = os.path.join(root_dir,
                                 'models',
                                 'model.ckpt-{}'.format(param.inference_iter))
        print('Using model {}'.format(model_str))
        saver.restore(sess, model_str)

        coord = tf.train.Coordinator()
        all_saved_flag = False
        try:
            seg_batch_runner.run_threads(sess, coord, num_threads=1)
            all_saved_flag=application.inference_loop(sess, coord, net_out)

        except KeyboardInterrupt:
            print('User cancelled training')
        except tf.errors.OutOfRangeError as e:
            pass
        except Exception:
            import sys
            import traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=sys.stdout)
        finally:
            if not all_saved_flag:
                print('stopped early, incomplete predictions')
            print('inference.py time: {:.3f} seconds'.format(
                time.time() - start_time))
            seg_batch_runner.close_all()
