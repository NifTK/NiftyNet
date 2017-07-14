# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import os.path
import time

import numpy as np
import scipy
import tensorflow as tf
from six.moves import range

from niftynet.engine.grid_sampler import GridSampler
from niftynet.engine.resize_sampler import ResizeSampler
from niftynet.engine.input_buffer import DeployInputBuffer
from niftynet.utilities.input_placeholders import ImagePatch
from niftynet.layer.post_processing import PostProcessingLayer

# run on single GPU with single thread
def run(net_class, param, volume_loader, device_str):
    # construct graph
    graph = tf.Graph()
    with graph.as_default(), tf.device("/{}:0".format(device_str)):
        # construct inference queue and graph
        # TODO change batch size param - batch size could be larger in test case

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

        # `patch` instance with image data only
        spatial_rank = patch_holder.spatial_rank
        if param.window_sampling in ['uniform', 'selective']:
            sampling_grid_size = param.label_size - 2 * param.border
            assert sampling_grid_size > 0
            sampler = GridSampler(patch=patch_holder,
                                  volume_loader=volume_loader,
                                  grid_size=sampling_grid_size,
                                  name='grid_sampler')
        elif param.window_sampling in ['resize']:
            sampler = ResizeSampler(
                patch=patch_holder,
                volume_loader=volume_loader,
                data_augmentation_methods=None,
                name="resize_sampler")
        net = net_class(num_classes=param.num_classes,
                        acti_func=param.activation_function)
        # construct train queue
        seg_batch_runner = DeployInputBuffer(
            batch_size=param.batch_size,
            capacity=max(param.queue_length, param.batch_size),
            sampler=sampler)
        test_pairs = seg_batch_runner.pop_batch_op
        info = test_pairs['info']
        logits = net(test_pairs['images'], is_training=False)

        # converting logits into final output for
        # classification probabilities or argmax classification labels
        if param.output_prob and param.num_classes > 1:
            post_process_layer = PostProcessingLayer(
                'SOFTMAX', num_classes=param.num_classes)
        elif not param.output_prob and param.num_classes > 1:
            post_process_layer = PostProcessingLayer(
                'ARGMAX', num_classes=param.num_classes)
        else:
            post_process_layer = PostProcessingLayer(
                'IDENTITY', num_classes=param.num_classes)
        net_out = post_process_layer(logits)
        variable_averages = tf.train.ExponentialMovingAverage(0.9)
        saver = tf.train.Saver()
        # variables_to_restore = variable_averages.variables_to_restore()
        # saver = tf.train.Saver(var_list=variables_to_restore)
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

        def nii_save(array, filename, output_dir=param.save_seg_dir):
            import nibabel as nib
            import sys
            new_image = nib.Nifti1Image(array, affine=np.eye(4))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            file_path = output_dir + '/' + filename + '.nii'
            if sys.platform == "win32":
                file_path.replace('/', '\\')
            nib.save(new_image, file_path)

        try:
            seg_batch_runner.run_threads(sess, coord, num_threads=1)

            if param.vae_inference_application == 'forward_pass':
                # Pass each volume through the network and save the reconstructions
                img_id, pred_img, subject_i = None, None, None
                while True:
                    local_time = time.time()
                    if coord.should_stop():
                        break
                    seg_maps, spatial_info = sess.run([net_out, info])
                    seg_maps = seg_maps[2]  # These are the data means
                    # go through each one in a batch
                    for batch_id in range(seg_maps.shape[0]):
                        img_id = spatial_info[batch_id, 0]
                        subject_i = volume_loader.get_subject(img_id)
                        predictions = seg_maps[batch_id]
                        predictions = np.expand_dims(predictions, axis = -1)  # ...NII requirements...
                        subject_i.save_network_output(
                            predictions,
                            param.save_seg_dir,
                            param.output_interp_order)
                        if patch_holder.is_stopping_signal(
                                spatial_info[batch_id]):
                            print('received finishing batch')
                            all_saved_flag = True
                            break
                all_saved_flag = True
            elif param.vae_inference_application == 'sample':
                # Generate one batch of samples from the prior, decode it, and save
                local_time = time.time()
                variance = float(param.linear_interpolation_variance)
                noise = np.random.normal(0, variance, (param.batch_size, net.number_of_latent_variables))
                dictionary_tmp = {logits[-1]: noise}.copy()
                current_decoded_interpolated_code = logits[2].eval(feed_dict=dictionary_tmp)
                for p in range(0, current_decoded_interpolated_code.shape[0]):
                    nii_save(current_decoded_interpolated_code[p,:,:,:,0], 'DecodedSampleFromThePrior_' + str(p))
                all_saved_flag = True
            elif param.vae_inference_application == 'linear_interpolation':
                # In the code space, linearly interpolate between the encodings of the first two volumes in the queue,
                # decode the interpolation steps, and save
                net_output = sess.run(net_out)
                real_codes = net_output[-1]
                originals = net_output[4]
                line = np.reshape(np.linspace(0, 1, num=param.batch_size), (param.batch_size,1))
                lin_interp_codes = line * (real_codes[1,:] - real_codes[0,:]) + real_codes[1,:]
                dictionary_tmp = {logits[-1]: lin_interp_codes}.copy()
                current_decoded_interpolated_code = logits[2].eval(feed_dict=dictionary_tmp)
                nii_save(originals[0, :, :, :, :], 'LinearInterpolationOriginalStart')
                for p in range(0,current_decoded_interpolated_code.shape[0]):
                    nii_save(current_decoded_interpolated_code[p, :, :, :, :], 'LinearInterpolation_' + str(p))
                nii_save(originals[1, :, :, :, :], 'LinearInterpolationOriginalFinish')
                all_saved_flag = True
            else:
                print('ERROR: the only VAE inference options are (1) forward_pass; (2) sample; and (3) linear_interpolation')



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
            seg_batch_runner.close_all()
        finally:
            if not all_saved_flag:
                print('stopped early, incomplete predictions')
            print('inference.py time: {:.3f} seconds'.format(
                time.time() - start_time))
            seg_batch_runner.close_all()
