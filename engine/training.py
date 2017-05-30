# -*- coding: utf-8 -*-
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import range

from utilities.filename_matching import KeywordsMatching
import utilities.misc_csv as misc_csv
from layer.input_buffer import TrainEvalInputBuffer
from layer.input_normalisation import HistogramNormalisationLayer as HistNorm
from utilities.input_placeholders import ImagePatch
from layer.loss import LossFunction
from layer.uniform_sampler import UniformSampler
from layer.volume_loader import VolumeLoaderLayer
from utilities import misc_common as util
from utilities.csv_table import CSVTable

np.random.seed(seed=int(time.time()))


def run(net_class, param, device_str):
    assert (param.batch_size <= param.queue_length)
    constraint_T1 = KeywordsMatching(
        ['./example_volumes/multimodal_BRATS'], ['T1'], ['T1c'])
    constraint_FLAIR = KeywordsMatching(
        ['./example_volumes/multimodal_BRATS'], ['Flair'], [])
    constraint_T1c = KeywordsMatching(
        ['./example_volumes/multimodal_BRATS'], ['T1c'], [])
    constraint_T2 = KeywordsMatching(
        ['./example_volumes/multimodal_BRATS'], ['T2'], [])
    constraint_array = [constraint_FLAIR,
                        constraint_T1,
                        constraint_T1c,
                        constraint_T2]
    misc_csv.write_matched_filenames_to_csv(
        constraint_array, './example_volumes/multimodal_BRATS/input.txt')

    constraint_Label = KeywordsMatching(
        ['./example_volumes/multimodal_BRATS'], ['Label'], [])
    misc_csv.write_matched_filenames_to_csv(
        [constraint_Label], './example_volumes/multimodal_BRATS/target.txt')

    csv_dict = {'input_image_file': './example_volumes/multimodal_BRATS/input.txt',
                'target_image_file': './example_volumes/multimodal_BRATS/target.txt',
                'weight_map_file': None,
                'target_note': None}

    # read each line of csv files into an instance of Subject
    csv_loader = CSVTable(csv_dict=csv_dict,
                          modality_names=('FLAIR', 'T1', 'T1c', 'T2'),
                          allow_missing=True)

    # define how to normalise image volumes
    hist_norm = HistNorm(
        models_filename=param.histogram_ref_file,
        multimod_mask_type=param.multimod_mask_type,
        norm_type=param.norm_type,
        cutoff=(param.cutoff_min, param.cutoff_max),
        mask_type=param.mask_type)
    # define how to choose training volumes
    spatial_padding = ((param.volume_padding_size, param.volume_padding_size),
                       (param.volume_padding_size, param.volume_padding_size),
                       (param.volume_padding_size, param.volume_padding_size))
    interp_order = (param.image_interp_order,
                    param.label_interp_order,
                    param.w_map_interp_order)
    param.reorientation = True if param.reorientation == "True" else False
    param.resampling = True if param.resampling == "True" else False
    param.normalisation = True if param.normalisation == "True" else False
    param.whitening = True if param.whitening == "True" else False
    volume_loader = VolumeLoaderLayer(
        csv_loader,
        hist_norm,
        is_training=True,
        do_reorientation= param.reorientation,
        do_resampling=param.resampling,
        spatial_padding=spatial_padding,
        do_normalisation=param.normalisation,
        do_whitening=param.whitening,
        interp_order=interp_order)
    print('found {} subjects'.format(len(volume_loader.subject_list)))

    graph = tf.Graph()
    with graph.as_default(), tf.device('/cpu:0'):
        # define a training element
        patch_holder = ImagePatch(
            image_shape=[param.image_size] * param.spatial_rank,
            label_shape=[param.label_size] * param.spatial_rank,
            weight_map_shape=[param.w_map_size] * param.spatial_rank,
            image_dtype=tf.float32,
            label_dtype=tf.int64,
            weight_map_dtype=tf.float32,
            num_image_modality=volume_loader.num_modality(0),
            num_label_modality=volume_loader.num_modality(1),
            num_weight_map=volume_loader.num_modality(2))

        # define how to generate samples from the volume
        augmentations = []
        if param.rotation == "True":
            from layer.rand_rotation import RandomRotationLayer
            augmentations.append(RandomRotationLayer(
                min_angle=param.min_angle,
                max_angle=param.max_angle))
        if param.spatial_scaling == "True":
            from layer.rand_spatial_scaling import RandomSpatialScalingLayer
            augmentations.append(RandomSpatialScalingLayer(
                max_percentage=param.max_percentage))

        sampler = UniformSampler(patch=patch_holder,
                                 volume_loader=volume_loader,
                                 patch_per_volume=param.sample_per_volume,
                                 data_augmentation_methods=augmentations,
                                 name='uniform_sampler')

        net = net_class(num_classes=param.num_classes)
        loss_func = LossFunction(n_class=param.num_classes,
                                 loss_type=param.loss_type,
                                 reg_type=param.reg_type,
                                 decay=param.decay)
        # construct train queue
        train_batch_runner = TrainEvalInputBuffer(batch_size=param.batch_size,
                                                  capacity=param.queue_length,
                                                  sampler=sampler,
                                                  shuffle=True)

        # optimizer
        train_step = tf.train.AdamOptimizer(learning_rate=param.lr)

        tower_misses = []
        tower_losses = []
        tower_grads = []
        train_pairs = train_batch_runner.pop_batch_op
        images = train_pairs['images']
        labels = train_pairs['labels']
        if "weight_maps" in train_pairs:
            weight_maps = train_pairs['weight_maps']
        else:
            weight_maps = None
        # _ = net(images, is_training=True)
        for i in range(0, param.num_gpus):
            with tf.device("/{}:{}".format(device_str, i)):
                predictions = net(images, is_training=True)
                loss = loss_func(predictions, labels, weight_maps)
                # TODO compute miss for dfferent target types
                miss = tf.reduce_mean(tf.cast(
                    tf.not_equal(tf.argmax(predictions, -1), labels[..., 0]),
                    dtype=tf.float32))

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
        summaries = [tf.summary.scalar("total-miss", ave_miss),
                     tf.summary.scalar("total-loss", ave_loss)]

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

    import pdb; pdb.set_trace()
    start_time = time.time()
    with tf.Session(config=config, graph=graph) as sess:
        # prepare output directory
        if not os.path.exists(param.model_dir + '/models'):
            os.makedirs(param.model_dir + '/models')
        root_dir = os.path.abspath(param.model_dir)
        # start or load session
        ckpt_name = root_dir + '/models/model.ckpt'
        if param.starting_iter > 0:
            model_str = ckpt_name + '-%d' % param.starting_iter
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
                print('iter {:d}, loss={:.8f},'
                      'error_rate={:.8f} ({:.3f}s)'.format(
                    i, loss_value, miss_value, time.time() - local_time))
                current_iter = i + param.starting_iter
                if (current_iter % 20) == 0:
                    writer.add_summary(sess.run(write_summary_op), current_iter)
                if (current_iter % param.save_every_n) == 0:
                    saver.save(sess, ckpt_name, global_step=current_iter)
                    print('Iter {} model saved at {}'.format(
                        i + param.starting_iter, ckpt_name))
        except KeyboardInterrupt:
            print('User cancelled training')
        except tf.errors.OutOfRangeError as e:
            print(e)
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
