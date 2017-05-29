# -*- coding: utf-8 -*-
import os
import os.path
import time

import numpy as np
import tensorflow as tf
from six.moves import range

import utilities.constraints_classes as cc
import utilities.misc_csv as misc_csv
from layer.grid_sampler import GridSampler
from layer.input_buffer import DeployInputBuffer
from layer.input_normalisation import HistogramNormalisationLayer as HistNorm
from layer.volume_loader import VolumeLoaderLayer
from utilities.csv_table import CSVTable
from utilities.input_placeholders import ImagePatch


# run on single GPU with single thread
def run(net_class, param, device_str):
    param_n_channel_out = 1
    param_output_interp_order = 3
    assert (param.batch_size <= param.queue_length)
    constraint_T1 = cc.ConstraintSearch(
        ['./example_volumes/multimodal_BRATS'], ['T1'], ['T1c'], ['_'])
    constraint_FLAIR = cc.ConstraintSearch(
        ['./example_volumes/multimodal_BRATS'], ['Flair'], [], ['_'])
    constraint_T1c = cc.ConstraintSearch(
        ['./example_volumes/multimodal_BRATS'], ['T1c'], [], ['_'])
    constraint_T2 = cc.ConstraintSearch(
        ['./example_volumes/multimodal_BRATS'], ['T2'], [], ['_'])
    constraint_array = [constraint_FLAIR,
                        constraint_T1,
                        constraint_T1c,
                        constraint_T2]
    misc_csv.write_matched_filenames_to_csv(
        constraint_array, './example_volumes/multimodal_BRATS/input.txt')

    constraint_Label = cc.ConstraintSearch(
        ['./example_volumes/multimodal_BRATS'], ['Label'], [], [])
    misc_csv.write_matched_filenames_to_csv(
        [constraint_Label], './example_volumes/multimodal_BRATS/target.txt')

    csv_dict = {
        'input_image_file': './example_volumes/multimodal_BRATS/input.txt',
        'target_image_file': None,
        'weight_map_file': None,
        'target_note': None}

    # read each line of csv files into an instance of Subject
    csv_loader = CSVTable(csv_dict=csv_dict,
                          modality_names=('FLAIR', 'T1', 'T1c', 'T2'),
                          allow_missing=True)

    # define how to normalise image volumes
    hist_norm = HistNorm(models_filename=param.histogram_ref_file,
                         multimod_mask_type=param.multimod_mask_type,
                         norm_type=param.norm_type,
                         cutoff=[x for x in param.norm_cutoff],
                         mask_type=param.mask_type)
    # define how to choose training volumes
    spatial_padding = ((param.volume_padding_size, param.volume_padding_size),
                       (param.volume_padding_size, param.volume_padding_size),
                       (param.volume_padding_size, param.volume_padding_size))

    volume_loader = VolumeLoaderLayer(csv_loader,
                                      hist_norm,
                                      is_training=False,
                                      do_reorientation=True,
                                      do_resampling=True,
                                      spatial_padding=spatial_padding,
                                      do_normalisation=True,
                                      do_whitening=True,
                                      interp_order=(3, 0))
    print('found {} subjects'.format(len(volume_loader.subject_list)))

    # construct graph
    graph = tf.Graph()
    with graph.as_default(), tf.device("/{}:0".format(device_str)):
        # construct inference queue and graph
        # TODO change batch size param - batch size could be larger in test case

        #patch_holder = ImagePatch(
        #    image_shape=[param.image_size] * 3,
        #    label_shape=None,
        #    weight_map_shape=None,
        #    image_dtype=tf.float32,
        #    label_dtype=tf.int64,
        #    weight_map_dtype=tf.float32,
        #    num_image_modality=volume_loader.num_modality(0),
        #    num_label_modality=volume_loader.num_modality(1),
        #    num_weight_map=1)

        # `patch` instance with image data only
        patch_holder = ImagePatch(
            image_shape=[param.image_size] * 3,
            image_dtype=tf.float32,
            num_image_modality=volume_loader.num_modality(0))
        spatial_rank = patch_holder.spatial_rank
        sampling_grid_size = patch_holder.image_size - 2 * param.border
        assert sampling_grid_size > 0
        sampler = GridSampler(patch=patch_holder,
                              volume_loader=volume_loader,
                              grid_size=sampling_grid_size,
                              name='grid_sampler')

        net = net_class(num_classes=param.num_classes)
        # construct train queue
        seg_batch_runner = DeployInputBuffer(batch_size=param.batch_size,
                                             capacity=param.queue_length,
                                             sampler=sampler)
        test_pairs = seg_batch_runner.pop_batch_op
        info = test_pairs['info']
        logits = net(test_pairs['images'], is_training=False)
        logits = tf.argmax(logits, -1)
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
        all_saved_flag = False
        try:
            seg_batch_runner.run_threads(sess, coord, num_threads=1)
            img_id, pred_img = None, None
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
                            subject_i.save_network_output(
                                pred_img,
                                param.save_seg_dir,
                                param_output_interp_order)
                        if patch_holder.is_stopping_signal(
                                spatial_info[batch_id]):
                            print('received finishing batch')
                            all_saved_flag = True
                            seg_batch_runner.close_all()

                        img_id = spatial_info[batch_id, 0]
                        subject_i = volume_loader.get_subject(img_id)
                        pred_img = subject_i.matrix_like_input_data_5d(
                            spatial_rank, param_n_channel_out)

                    # try to expand prediction dims to match the output volume
                    predictions = seg_maps[batch_id]
                    while predictions.ndim < pred_img.ndim:
                        predictions = np.expand_dims(predictions, axis=-1)

                    # assign predicted patch to the allocated output volume
                    origin = spatial_info[batch_id, 1:(1+spatial_rank)]
                    # indexing within the patch
                    s_ = param.border
                    _s = patch_holder.image_size - param.border
                    # indexing within the prediction volume
                    dest_start = origin + s_
                    dest_end = origin + _s
                    assert np.all(dest_start >= 0)
                    assert np.all(dest_end <= pred_img.shape[0:spatial_rank])
                    if spatial_rank == 3:
                        x_, y_, z_ = dest_start
                        _x, _y, _z = dest_end
                        pred_img[x_:_x, y_:_y, z_:_z, ...] = \
                            predictions[s_:_s, s_:_s, s_:_s, ...]
                    elif spatial_rank == 2:
                        x_, y_ = dest_start
                        _x, _y = dest_end
                        pred_img[x_:_x, y_:_y, ...] = \
                            predictions[s_:_s, s_:_s, ...]
                    else:
                        raise ValueError("unsupported spatial rank")

        except KeyboardInterrupt:
            print('User cancelled training')
        except tf.errors.OutOfRangeError:
            pass
        except Exception as unusual_error:
            print(unusual_error)
            seg_batch_runner.close_all()
        finally:
            if not all_saved_flag:
                raise ValueError('stopped early, incomplete predictions')
            print('inference.py time: {:.3f} seconds'.format(
                time.time() - start_time))
            seg_batch_runner.close_all()
