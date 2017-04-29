# -*- coding: utf-8 -*-
import os
import os.path
import time

import numpy as np
import tensorflow as tf
from six.moves import range

import utilities.misc as util
from network.net_template import NetTemplate
from input_queue import DeployInputBuffer
from sampler import VolumeSampler


# run on single GPU with single thread
def run(net, param):
    if not isinstance(net, NetTemplate):
        print('Net model should inherit from NetTemplate')
        return

    valid_names = util.list_patient(param.eval_data_dir)
    mod_list = util.list_modality(param.train_data_dir)
    rand_sampler = VolumeSampler(valid_names,
                                 mod_list,
                                 net.batch_size,
                                 net.input_image_size,
                                 net.input_label_size,
                                 param.volume_padding_size,
                                 param.histogram_ref_file)
    sampling_grid_size = net.input_label_size - 2 * param.border
    if sampling_grid_size <= 0:
        print('Param error: non-positive sampling grid_size')
        return None
    sample_generator = rand_sampler.grid_sampling_from(
        param.eval_data_dir, sampling_grid_size, yield_seg=False)

    # construct graph
    graph = tf.Graph()
    with graph.as_default(), tf.device("/gpu:0"):  # TODO multiple GPU?
        # construct train queue and graph
        # TODO change batch size param - batch size could be larger in test case
        seg_batch_runner = DeployInputBuffer(
            net.batch_size,
            param.queue_length,
            shapes=[[net.input_image_size] * 3 + [len(mod_list)], [7]],
            sample_generator=sample_generator)
        test_pairs = seg_batch_runner.pop_batch()
        info = test_pairs['info']
        logits = net.inference(test_pairs['images'])
        logits = tf.argmax(logits, 4)
        variable_averages = tf.train.ExponentialMovingAverage(0.9)
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
        ckpt = tf.train.get_checkpoint_state(root_dir + '/models/')
        if ckpt and ckpt.model_checkpoint_path:
            print('Evaluation from checkpoints')
        model_str = '{}/models/model.ckpt-{}'.format(root_dir, param.pred_iter)
        print('Using model {}'.format(model_str))
        saver.restore(sess, model_str)

        coord = tf.train.Coordinator()
        try:
            seg_batch_runner.init_threads(sess, coord, num_threads=1)
            img_id = -1
            pred_img = None
            patient_name = None
            while True:
                if coord.should_stop():
                    break
                seg_maps, spatial_info = sess.run([logits, info])
                # go through each one in a batch
                for batch_id in range(seg_maps.shape[0]):
                    if spatial_info[batch_id, 0] != img_id:
                        # when loc_info changed
                        # save current map and reset cumulative map variable
                        if pred_img is not None:
                            util.save_segmentation(
                                param, patient_name, pred_img)
                        img_id = spatial_info[batch_id, 0]
                        patient_name = valid_names[img_id]
                        pred_img = util.volume_of_zeros_like(
                            param.eval_data_dir, patient_name, mod_list[0])
                        pred_img = np.pad(
                            pred_img, param.volume_padding_size, 'minimum')
                        #print('init %s' % valid_names[img_id])
                    loc_x = spatial_info[batch_id, 1]
                    loc_y = spatial_info[batch_id, 2]
                    loc_z = spatial_info[batch_id, 3]

                    p_start = param.border
                    p_end = net.input_label_size - param.border
                    predictions = seg_maps[batch_id]
                    pred_img[(loc_x + p_start): (loc_x + p_end),
                             (loc_y + p_start): (loc_y + p_end),
                             (loc_z + p_start): (loc_z + p_end)] = \
                        predictions[p_start: p_end,
                                    p_start: p_end,
                                    p_start: p_end]

        except KeyboardInterrupt:
            print('User cancelled training')
        except tf.errors.OutOfRangeError:
            pass
        # except Exception as unusual_error:
        #    print(unusual_error)
        #    seg_batch_runner.close_all(coord, sess)
        finally:
            # save the last batches
            util.save_segmentation(param, valid_names[img_id], pred_img)
            print('inference.py time: {} seconds'.format(
                time.time() - start_time))
            seg_batch_runner.close_all(coord, sess)
            coord.join(seg_batch_runner.threads)
