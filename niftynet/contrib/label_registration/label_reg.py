import tensorflow as tf
import h5py
import numpy as np
import os
import sys
import random
import time
import tensorflowhelpers as tfhelpers
import tensorflowtools as tftools

flag_randomHyperParam = False

# architecture parameter
num_channel_initial = 32
conv_size_initial = 5
num_resolution_level = 4

# algorithm
learning_rate = 1e-5
lambda_decay = 0
lambda_bending = 1e-3
lambda_gradient = 1e-3

# data set
dataset_name_target = 'test1-us1'
dataset_name_moving = 'test1-mr1'

# training
initial_bias = 0.01
totalIterations = 10001
miniBatchSize = 8

# log and data saving
log_num_shuffle = 0
log_start_debug = 100
log_freq_debug = 100
log_freq_info = 10

# --- parameters to be read from file
working_size = []
size_ddf = []

flag_single_label = True
flag_use_mask = True  # in with long-tailed postprocessing


# k-fold cross-validation
num_fold = 10  # k
idx_fold = 0

# --- experimental sampling --- # BEFORE seeding!
if flag_randomHyperParam:
    miniBatch_size = int(np.random.choice([16, 25, 36, 49, 64, 100]))
    # devCase_size = num_case-1
    # noise_size = int(np.random.choice([1e2, 500]))
    # idx_ROI = int(np.random.choice([0, 1, 2, 3, 4]))
    idx_crossV = int(np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
    num_channel_initial_G = int(np.random.choice([128, 256, 512, 1024]))
    num_channel_initial_D = int(np.random.choice([8, 16, 32, 64]))
    lambda_weight_G = np.random.choice([0, 1e-3, 1e-5])
    lambda_weight_D = np.random.choice([0, 1e-3, 1e-5])
    conv_kernel_size_initial = int(np.random.choice([3, 5, 7]))
    lambda_supervised = np.random.choice([0, 1e-2, 1e-3, 1e-4, 1e-5])
    order_supervised = np.random.choice([1, 2])
    # md_num_features = int(np.random.choice([100, 200, 300, 400, 500]))
    # useConditionSmoother = int(np.random.choice([True, False]))
    learning_rate = np.random.choice([1e-4, 2e-4, 5e-4])
# --- experimental sampling ---

# data sets
h5fn_image_target, h5fn_label_target, size_target, _, h5fn_mask_target = tfhelpers.dataset_switcher(dataset_name_target)
h5fn_image_moving, h5fn_label_moving, size_moving, totalDataSize, _ = tfhelpers.dataset_switcher(dataset_name_moving)

# logging
filename_output_info = 'output_info.txt'
dir_output = os.path.join(os.environ['HOME'], 'Scratch/output/labelreg0/', "%f" % time.time())
flag_dir_overwrite = os.path.exists(dir_output)
os.makedirs(dir_output)
fid_output_info = open(os.path.join(dir_output, filename_output_info), 'a')
if flag_dir_overwrite:
    print('\nWARNING: %s existed - files may be overwritten.\n\n' % dir_output)
    print('\nWARNING: %s existed - files may be overwritten.\n\n' % dir_output, flush=True, file=fid_output_info)


# information
print('- Algorithm Summary (normalised calibration) --------', flush=True, file=fid_output_info)

print('current_time: %s' % time.asctime(time.gmtime()), flush=True, file=fid_output_info)
print('flag_dir_overwrite: %s' % flag_dir_overwrite, flush=True, file=fid_output_info)
print('idx_ROI: %s' % idx_ROI, flush=True, file=fid_output_info)
print('idx_crossV: %s' % idx_crossV, flush=True, file=fid_output_info)
print('data_set_name: %s' % data_set_name, flush=True, file=fid_output_info)
print('cond_set_name: %s' % cond_set_name, flush=True, file=fid_output_info)
print('miniBatch_size: %s' % miniBatch_size, flush=True, file=fid_output_info)
print('noise_size: %s' % noise_size, flush=True, file=fid_output_info)
print('conv_kernel_size_initial: %s' % conv_kernel_size_initial, flush=True, file=fid_output_info)
print('num_channel_initial_G: %s' % num_channel_initial_G, flush=True, file=fid_output_info)
print('num_channel_initial_D: %s' % num_channel_initial_D, flush=True, file=fid_output_info)
print('lambda_weight_G: %s' % lambda_weight_G, flush=True, file=fid_output_info)
print('lambda_weight_D: %s' % lambda_weight_D, flush=True, file=fid_output_info)
print('lambda_supervised: %s' % lambda_supervised, flush=True, file=fid_output_info)
print('order_supervised: %s' % order_supervised, flush=True, file=fid_output_info)
print('learning_rate: %s' % learning_rate, flush=True, file=fid_output_info)
print('md_num_features: %s' % md_num_features, flush=True, file=fid_output_info)
print('md_num_kernels: %s' % md_num_kernels, flush=True, file=fid_output_info)
print('md_kernel_dim: %s' % md_kernel_dim, flush=True, file=fid_output_info)
print('keep_prob_rate: %s' % keep_prob_rate, flush=True, file=fid_output_info)
print('generator_shortcuts: %s' % generator_shortcuts, flush=True, file=fid_output_info)
print('useIdentityMapping: %s' % useIdentityMapping, flush=True, file=fid_output_info)
print('useConditionSmoother: %s' % useConditionSmoother, flush=True, file=fid_output_info)

print('- End of Algorithm Summary --------', flush=True, file=fid_output_info)
if log_num_shuffle:
    print('trainDataIndices: %s' % trainDataIndices, flush=True, file=fid_output_info)


# totalDataSize = 111
random.seed(1)
dataIndices = [i for i in range(totalDataSize)]
random.shuffle(dataIndices)  # shuffle once
foldSize = int(totalDataSize / num_fold)
testIndices = [dataIndices[i] for i in range(foldSize*idx_fold, foldSize*(idx_fold+1))]

# grouping the remainders to test
remainder = totalDataSize % num_fold
if (remainder != 0) & (idx_fold < remainder):
    testIndices.append(dataIndices[totalDataSize-remainder+idx_fold])

trainIndices = list(set(dataIndices) - set(testIndices))
random.shuffle(trainIndices)
trainSize = len(trainIndices)

# setting up minibatch gradient descent

num_miniBatch = int(trainSize / miniBatchSize)
# for inference using minibatch
remainder = len(testIndices) % miniBatchSize
if remainder != 0:
    testIndices += [dataIndices[i] for i in range(miniBatchSize-remainder)]
num_miniBatch_test = int(len(testIndices) / miniBatchSize)


# pre-computing for graph
grid_reference = tftools.get_reference_grid(size_target)

# building the computational graph
movingImage_ph = tf.placeholder(tf.float32, [miniBatchSize]+size_moving+[1])
targetImage_ph = tf.placeholder(tf.float32, [miniBatchSize]+size_target+[1])
movingLabel_ph = tf.placeholder(tf.float32, [miniBatchSize]+size_moving+[1])
targetLabel_ph = tf.placeholder(tf.float32, [miniBatchSize]+size_target+[1])
movingTransform_ph = tf.placeholder(tf.float32, [miniBatchSize]+[1, 12])
targetTransform_ph = tf.placeholder(tf.float32, [miniBatchSize]+[1, 12])
# keep_prob_ph = tf.placeholder(tf.float32)

# random spatial transform - do not rescaling moving here
movingImage0 = tftools.random_transform1(movingImage_ph, movingTransform_ph, size_moving)
movingLabel0 = tftools.random_transform1(movingLabel_ph, movingTransform_ph, size_moving)
targetImage0 = tftools.random_transform1(targetImage_ph, targetTransform_ph, size_target)
targetLabel0 = tftools.random_transform1(targetLabel_ph, targetTransform_ph, size_target)

strides_none = [1, 1, 1, 1, 1]
strides_down = [1, 2, 2, 2, 1]
k_conv = [3, 3, 3]
k_conv0 = [conv_size_initial, conv_size_initial, conv_size_initial]
k_pool = [1, 2, 2, 2, 1]


# down-sampling
nc0 = num_channel_initial
W0c = tf.get_variable("W0c", shape=k_conv0+[2, nc0], initializer=tf.contrib.layers.xavier_initializer())
W0r1 = tf.get_variable("W0r1", shape=k_conv+[nc0, nc0], initializer=tf.contrib.layers.xavier_initializer())
W0r2 = tf.get_variable("W0r2", shape=k_conv+[nc0, nc0], initializer=tf.contrib.layers.xavier_initializer())
h0c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(
    tf.concat([tftools.resize_volume(movingImage0, size_target), targetImage0], axis=4), W0c, strides_none, "SAME")))
h0r1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0c, W0r1, strides_none, "SAME")))
h0r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0r1, W0r2, strides_none, "SAME")) + h0c)
h0 = tf.nn.max_pool3d(h0r2, k_pool, strides_down, padding="SAME")
theta = [W0c, W0r1, W0r2]

nc1 = nc0*2
W1c = tf.get_variable("W1c", shape=k_conv+[nc0, nc1], initializer=tf.contrib.layers.xavier_initializer())
W1r1 = tf.get_variable("W1r1", shape=k_conv+[nc1, nc1], initializer=tf.contrib.layers.xavier_initializer())
W1r2 = tf.get_variable("W1r2", shape=k_conv+[nc1, nc1], initializer=tf.contrib.layers.xavier_initializer())
theta += [W1c, W1r1, W1r2]
h1c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0, W1c, strides_none, "SAME")))
h1r1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1c, W1r1, strides_none, "SAME")))
h1r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1r1, W1r2, strides_none, "SAME")) + h1c)
h1 = tf.nn.max_pool3d(h1r2, k_pool, strides_down, padding="SAME")

nc2 = nc1*2
W2c = tf.get_variable("W2c", shape=k_conv+[nc1, nc2], initializer=tf.contrib.layers.xavier_initializer())
W2r1 = tf.get_variable("W2r1", shape=k_conv+[nc2, nc2], initializer=tf.contrib.layers.xavier_initializer())
W2r2 = tf.get_variable("W2r2", shape=k_conv+[nc2, nc2], initializer=tf.contrib.layers.xavier_initializer())
theta += [W2c, W2r1, W2r2]
h2c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1, W2c, strides_none, "SAME")))
h2r1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2c, W2r1, strides_none, "SAME")))
h2r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2r1, W2r2, strides_none, "SAME")) + h2c)
h2 = tf.nn.max_pool3d(h2r2, k_pool, strides_down, padding="SAME")

nc3 = nc2*2
W3c = tf.get_variable("W3c", shape=k_conv+[nc2, nc3], initializer=tf.contrib.layers.xavier_initializer())
W3r1 = tf.get_variable("W3r1", shape=k_conv+[nc3, nc3], initializer=tf.contrib.layers.xavier_initializer())
W3r2 = tf.get_variable("W3r2", shape=k_conv+[nc3, nc3], initializer=tf.contrib.layers.xavier_initializer())
theta += [W3c, W3r1, W3r2]
h3c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2, W3c, strides_none, "SAME")))
h3r1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3c, W3r1, strides_none, "SAME")))
h3r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3r1, W3r2, strides_none, "SAME")) + h3c)
h3 = tf.nn.max_pool3d(h3r2, k_pool, strides_down, padding="SAME")

# deep
ncD = nc3*2  # deep layer
WD = tf.get_variable("WD", shape=k_conv+[nc3, ncD], initializer=tf.contrib.layers.xavier_initializer())
theta += [WD]
hD = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3, WD, strides_none, "SAME")))

# up-sampling
W3_c = tf.get_variable("W3_c", shape=k_conv+[nc3, ncD], initializer=tf.contrib.layers.xavier_initializer())
W3_r1 = tf.get_variable("W3_r1", shape=k_conv+[nc3, nc3], initializer=tf.contrib.layers.xavier_initializer())
W3_r2 = tf.get_variable("W3_r2", shape=k_conv+[nc3, nc3], initializer=tf.contrib.layers.xavier_initializer())
theta += [W3_c, W3_r1, W3_r2]
h3_c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(hD, W3_c, h3c.get_shape(), strides_down, "SAME")))
h3_r1 = tf.add(h3_c, h3c)
h3_r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3_r1, W3_r1, strides_none, "SAME")))
h3_ = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3_r2, W3_r2, strides_none, "SAME")) + h3_r1)

W2_c = tf.get_variable("W2_c", shape=k_conv+[nc2, nc3], initializer=tf.contrib.layers.xavier_initializer())
W2_r1 = tf.get_variable("W2_r1", shape=k_conv+[nc2, nc2], initializer=tf.contrib.layers.xavier_initializer())
W2_r2 = tf.get_variable("W2_r2", shape=k_conv+[nc2, nc2], initializer=tf.contrib.layers.xavier_initializer())
theta += [W2_c, W2_r1, W2_r2]
h2_c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(h3_, W2_c, h2c.get_shape(), strides_down, "SAME")))
h2_r1 = tf.add(h2_c, h2c)
h2_r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2_r1, W2_r1, strides_none, "SAME")))
h2_ = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2_r2, W2_r2, strides_none, "SAME")) + h2_r1)

W1_c = tf.get_variable("W1_c", shape=k_conv+[nc1, nc2], initializer=tf.contrib.layers.xavier_initializer())
W1_r1 = tf.get_variable("W1_r1", shape=k_conv+[nc1, nc1], initializer=tf.contrib.layers.xavier_initializer())
W1_r2 = tf.get_variable("W1_r2", shape=k_conv+[nc1, nc1], initializer=tf.contrib.layers.xavier_initializer())
theta += [W1_c, W1_r1, W1_r2]
h1_c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(h2_, W1_c, h1c.get_shape(), strides_down, "SAME")))
h1_r1 = tf.add(h1_c, h1c)
h1_r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1_r1, W1_r1, strides_none, "SAME")))
h1_ = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1_r2, W1_r2, strides_none, "SAME")) + h1_r1)

W0_c = tf.get_variable("W0_c", shape=k_conv+[nc0, nc1], initializer=tf.contrib.layers.xavier_initializer())
W0_r1 = tf.get_variable("W0_r1", shape=k_conv+[nc0, nc0], initializer=tf.contrib.layers.xavier_initializer())
W0_r2 = tf.get_variable("W0_r2", shape=k_conv+[nc0, nc0], initializer=tf.contrib.layers.xavier_initializer())
theta += [W0_c, W0_r1, W0_r2]
h0_c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(h1_, W0_c, h0c.get_shape(), strides_down, "SAME")))
h0_r1 = tf.add(h0_c, h0c)
h0_r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0_r1, W0_r1, strides_none, "SAME")))
h0_ = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0_r2, W0_r2, strides_none, "SAME")) + h0_r1)

# readout
nc_o = 3
W_o = tf.get_variable("W_o", shape=k_conv+[nc0, nc_o], initializer=tf.contrib.layers.xavier_initializer())
b_o = tf.Variable(tf.constant(initial_bias, shape=[nc_o]), name='b_o')
theta += [W_o]  # but bias term
displacement = tf.nn.conv3d(h0_, W_o, strides_none, "SAME") + b_o
grid_sample = grid_reference + displacement

# label-smoothed cross-entropy
movingLabel_warped = tftools.resample_linear(movingLabel0, grid_sample)
ce = tf.nn.softmax_cross_entropy_with_logits(labels=tf.concat([targetLabel0, 1-targetLabel0], axis=4),
                                             logits=tf.concat([movingLabel_warped, 1-movingLabel_warped], axis=4))
tf.add_to_collection('loss', tf.reduce_mean(ce))

# bending energy
if lambda_bending > 0:
    be_batches = tftools.compute_bending_energy(displacement)
    be = tf.reduce_mean(be_batches)
    tf.add_to_collection('loss', be * lambda_bending)
else:
    be = tf.constant(-1)

# gradient L2
if lambda_gradient > 0:
    gn_batches = tftools.compute_gradient_l2norm(displacement)
    gn = tf.reduce_mean(gn_batches)
    tf.add_to_collection('loss', gn * lambda_gradient)
else:
    gn = tf.constant(-1)

# weight-decay
if lambda_decay > 0:
    for i in range(len(theta)):
        tf.add_to_collection('loss', tf.nn.l2_loss(theta[i]) * lambda_decay)

# loss
loss = tf.add_n(tf.get_collection('loss'))


# branch nodes
# dice, movingVol, targetVol = tftools.compute_dice(movingLabel_warped, targetLabel0)
dice, _, _ = tftools.compute_dice(movingLabel_warped, targetLabel0)
dist = tftools.compute_centroid_distance(movingLabel_warped, targetLabel0, grid_reference)


# setting up optimisation
optimizer = tf.train.AdamOptimizer(learning_rate)
global_step = tf.Variable(0, name='global_step', trainable=False)  # step counter
train_op = optimizer.minimize(loss, global_step=global_step)
# train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# pre-compute
transform_identity = tfhelpers.initial_transform_generator(miniBatchSize)
feeder_target = tfhelpers.DataFeeder(h5fn_image_target, h5fn_label_target, h5fn_mask_target)
feeder_moving = tfhelpers.DataFeeder(h5fn_image_moving, h5fn_label_moving)

for step in range(totalIterations):
    current_time = time.asctime(time.gmtime())

    if step in range(0, totalIterations, num_miniBatch):
        random.shuffle(trainIndices)
        if step < num_miniBatch * log_num_shuffle:
            print('trainDataIndices: %s' % trainIndices, flush=True, file=fid_output_info)

    miniBatch_idx = step % num_miniBatch
    miniBatch_indices = trainIndices[miniBatch_idx*miniBatchSize:(miniBatch_idx + 1)*miniBatchSize]
    if step < num_miniBatch * log_num_shuffle:
        print('miniBatch_indices: %s' % miniBatch_indices, flush=True, file=fid_output_info)

    if flag_single_label:
        label_indices = [0] * miniBatchSize
    else:
        label_indices = [random.randrange(feeder_moving.num_labels[i]) for i in miniBatch_indices]

    trainFeed = {movingImage_ph: feeder_moving.get_image_batch(miniBatch_indices),
                 targetImage_ph: feeder_target.get_image_batch(miniBatch_indices),
                 movingLabel_ph: feeder_moving.get_label_batch(miniBatch_indices, label_indices),
                 targetLabel_ph: feeder_target.get_label_batch(miniBatch_indices, label_indices),
                 movingTransform_ph: tfhelpers.random_transform_generator(miniBatchSize),
                 targetTransform_ph: tfhelpers.random_transform_generator(miniBatchSize)}

    if step in range(0, totalIterations, log_freq_info):
        _, loss_train, be_train, gn_train, dice_train, dist_train = sess.run([train_op, loss, be, gn, dice, dist], feed_dict=trainFeed)
        print('[%s] Step %d: loss=%f, be=%f, gn=%f' % (current_time, step, loss_train, be_train, gn_train))
        print('[%s] Step %d: loss=%f, be=%f, gn=%f' % (current_time, step, loss_train, be_train, gn_train), flush=True, file=fid_output_info)
        print('  Dice: %s' % np.reshape(dice_train, newshape=[1, -1]))
        print('  Dice: %s' % np.reshape(dice_train, newshape=[1, -1]), flush=True, file=fid_output_info)
        print('  Distance: %s' % dist_train)
        print('  Distance: %s' % dist_train, flush=True, file=fid_output_info)
    else:
        sess.run(train_op, feed_dict=trainFeed)

    # Debug data
    if step in range(log_start_debug, totalIterations, log_freq_debug):
        # save debug samples
        filename_log_data = "debug_data_i%09d.h5" % step
        fid_debug_data = h5py.File(os.path.join(dir_output, filename_log_data), 'w')

        # --- choose the variables to save ---
        '''
        movingImage0_train, targetImage0_train = sess.run([movingImage0, targetImage0], feed_dict=trainFeed)
        fid_debug_data.create_dataset('/movingImage0_train/', movingImage0_train.shape, dtype=movingImage0_train.dtype, data=movingImage0_train)
        fid_debug_data.create_dataset('/targetImage0_train/', targetImage0_train.shape, dtype=targetImage0_train.dtype, data=targetImage0_train)
        grid_reference_train = sess.run(grid_reference, feed_dict=trainFeed)
        fid_debug_data.create_dataset('/grid_reference_train/', grid_reference_train.shape, dtype=grid_reference_train.dtype, data=grid_reference_train)
        '''
        movingLabel0_train, targetLabel0_train = sess.run([movingLabel0, targetLabel0], feed_dict=trainFeed)
        fid_debug_data.create_dataset('/movingLabel0_train/', movingLabel0_train.shape, dtype=movingLabel0_train.dtype, data=movingLabel0_train)
        fid_debug_data.create_dataset('/targetLabel0_train/', targetLabel0_train.shape, dtype=targetLabel0_train.dtype, data=targetLabel0_train)
        displacement_train, movingLabel_warped_train = sess.run([displacement, movingLabel_warped], feed_dict=trainFeed)
        fid_debug_data.create_dataset('/displacement_train/', displacement_train.shape, dtype=displacement_train.dtype, data=displacement_train)
        fid_debug_data.create_dataset('/movingLabel_warped_train/', movingLabel_warped_train.shape, dtype=movingLabel_warped_train.dtype, data=movingLabel_warped_train)
        # --------------------------

        # --- test ---
        fid_debug_data.create_dataset('/testIndices/', data=testIndices)
        for k in range(num_miniBatch_test):
            idx_test = [testIndices[i] for i in range(miniBatchSize*k, miniBatchSize*(k+1))]
            testFeed = {movingImage_ph: feeder_moving.get_image_batch(idx_test),
                        targetImage_ph: feeder_target.get_image_batch(idx_test),
                        movingLabel_ph: feeder_moving.get_label_batch(idx_test, label_indices),
                        targetLabel_ph: feeder_target.get_label_batch(idx_test, label_indices),
                        movingTransform_ph: transform_identity,
                        targetTransform_ph: transform_identity}
            displacement_test, movingLabel_warped_test = sess.run([displacement, movingLabel_warped], feed_dict=testFeed)
            fid_debug_data.create_dataset('/displacement_test_k%d/' % k, displacement_test.shape, dtype=displacement_test.dtype, data=displacement_test)
            fid_debug_data.create_dataset('/movingLabel_warped_test_k%d/' % k, movingLabel_warped_test.shape, dtype=movingLabel_warped_test.dtype, data=movingLabel_warped_test)

            dice_test, dist_test = sess.run([dice, dist], feed_dict=testFeed)
            print('test-Dice: %s' % np.reshape(dice_test, newshape=[1, -1]))
            print('test-Dice: %s' % np.reshape(dice_test, newshape=[1, -1]), flush=True, file=fid_output_info)
            print('test-Distance: %s' % dist_test)
            print('test-Distance: %s' % dist_test, flush=True, file=fid_output_info)
            fid_debug_data.create_dataset('/dice_test_k%d/' % k, data=dice_test)
            fid_debug_data.create_dataset('/dist_test_k%d/' % k, data=dist_test)
        # ------------

        # flush in the end
        fid_debug_data.flush()
        fid_debug_data.close()
        print('Debug data saved at Step %d' % step)
        print('Debug data saved at Step %d' % step, flush=True, file=fid_output_info)
        # NB. do not use continue here!
# ---------- End of Computational Graph ----------

fid_output_info.close()
