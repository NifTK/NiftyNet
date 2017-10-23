import tensorflow as tf
import h5py
import numpy as np
import os
# import sys
import random
import time
import helpers as tfhelpers
import utils as tftools

flag_randomHyperParam = False
flag_debugPrint = True

# architecture parameter
num_channel_initial = 32
conv_size_initial = 3
idx_size_ffd = 0  # this is using sizes in sizes_target_hierarchy, 0 being original size, nonzero sigma_filtering
num_channel_initial_global = 4

# algorithm
learning_rate = 1e-5
lambda_decay = 1e-6
lambda_bending = [1e-2]  # [1e-3, 1e-3, 1e-3, 1e-3]  # sizes_target_hierarchy
lambda_gradient = [0]  # [0, 0, 0, 0]  # sizes_target_hierarchy
sigma_filtering = 0  # in voxel not in ffd grid

# data set
dataset_name_target = 'test-1-us1'
dataset_name_moving = 'test-1-mr1'

# training
miniBatchSize = 10
flag_use_global = True
flag_use_local = True
start_composite = 2000  # only used when both global + local
flag_single_label = False
importance_sampling = 0.5
start_multiple_label = 0  # only used when flag_single_label==False
initial_bias = 0.0
initial_std_local = 1e-8
initial_std_global = 1e-8
label_smoothing = 0.1
totalIterations = 100001
initialiser_local = tf.random_normal_initializer(0, initial_std_local)  # tf.contrib.layers.xavier_initializer()
initialiser_global = tf.random_normal_initializer(0, initial_std_global)  # tf.contrib.layers.xavier_initializer()

# k-fold cross-validation
num_fold = 10  # k
idx_fold = 0

# log and data saving
log_num_shuffle = 0
log_start_debug = 5000
log_freq_debug = 1000
log_freq_info = 100


# --- experimental parameter searching --- # BEFORE seeding!
if flag_randomHyperParam:
    learning_rate = random.choice([1e-5, 1e-6])
    lambda_decay = random.choice([0, 1e-3, 1e-4, 1e-5, 1e-6])
    lambda_bending = [random.choice([1e-1, 5e-2, 1e-2]),
                      random.choice([0]),
                      random.choice([0]),
                      random.choice([0])]

    """
    lambda_gradient = [random.choice([0, 1, 0.5, 1e-1, 1e-2]),
                       random.choice([0]),
                       random.choice([0]),
                       random.choice([0])]
    
    flag_use_local = random.choice([True, False])
    if flag_use_local:
        flag_use_global = random.choice([True, False])
    else:
        flag_use_global = True    
    """

    idx_fold = random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    start_composite = random.choice([0, 1000, 2000])
    # start_multiple_label = random.choice([0, 1000])
    importance_sampling = random.choice([0.5, 0.66666667, 0.75])
    # dataset_name_target = random.choice(['test-1-us1', 'test1-us1', 'test3-us1', 'test5-us1'])
    # dataset_name_moving = random.choice(['test-1-mr1', 'test1-mr1', 'test3-mr1'])
    # sigma_filtering = random.choice([0, 1, 2, 3, 4, 5])
    conv_size_initial = random.choice([3, 5])

    # num_channel_initial_global = random.choice([4, 8])
    # num_channel_initial = random.choice([16, 32])
    if num_channel_initial == 32:
        miniBatchSize = 4
    elif num_channel_initial == 16:
        miniBatchSize = 6
    elif num_channel_initial == 8:
        miniBatchSize = 10
# --- experimental parameter searching ---

# start_multiple_label = start_multiple_label * (not flag_single_label)
start_composite = start_composite * flag_use_local * flag_use_global
flag_use_composite = flag_use_global & flag_use_local

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
print('- Algorithm Summary (i-net-4) --------', flush=True, file=fid_output_info)

print('current_time: %s' % time.asctime(time.gmtime()), flush=True, file=fid_output_info)
print('flag_dir_overwrite: %s' % flag_dir_overwrite, flush=True, file=fid_output_info)
print('num_channel_initial: %s' % num_channel_initial, flush=True, file=fid_output_info)
print('conv_size_initial: %s' % conv_size_initial, flush=True, file=fid_output_info)
print('idx_size_ffd: %s' % idx_size_ffd, flush=True, file=fid_output_info)
print('num_channel_initial_global: %s' % num_channel_initial_global, flush=True, file=fid_output_info)
print('learning_rate: %s' % learning_rate, flush=True, file=fid_output_info)
print('lambda_decay: %s' % lambda_decay, flush=True, file=fid_output_info)
print('lambda_bending: %s' % lambda_bending, flush=True, file=fid_output_info)
print('lambda_gradient: %s' % lambda_gradient, flush=True, file=fid_output_info)
print('sigma_filtering: %s' % sigma_filtering, flush=True, file=fid_output_info)
print('flag_single_label: %s' % flag_single_label, flush=True, file=fid_output_info)
print('dataset_name_target: %s' % dataset_name_target, flush=True, file=fid_output_info)
print('dataset_name_moving: %s' % dataset_name_moving, flush=True, file=fid_output_info)
print('initial_bias: %s' % initial_bias, flush=True, file=fid_output_info)
print('initial_std_global: %s' % initial_std_global, flush=True, file=fid_output_info)
print('initial_std_local: %s' % initial_std_local, flush=True, file=fid_output_info)
print('label_smoothing: %s' % label_smoothing, flush=True, file=fid_output_info)
print('totalIterations: %s' % totalIterations, flush=True, file=fid_output_info)
print('flag_use_global: %s' % flag_use_global, flush=True, file=fid_output_info)
print('flag_use_local: %s' % flag_use_local, flush=True, file=fid_output_info)
print('start_composite: %s' % start_composite, flush=True, file=fid_output_info)
print('start_multiple_label: %s' % start_multiple_label, flush=True, file=fid_output_info)
print('importance_sampling: %s' % importance_sampling, flush=True, file=fid_output_info)
print('miniBatchSize: %s' % miniBatchSize, flush=True, file=fid_output_info)
print('num_fold: %s' % num_fold, flush=True, file=fid_output_info)
print('idx_fold: %s' % idx_fold, flush=True, file=fid_output_info)

print('- End of Algorithm Summary --------', flush=True, file=fid_output_info)

random.seed(1)
tf.set_random_seed(1)

# totalDataSize = 111
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
if log_num_shuffle:
    print('trainDataIndices: %s' % trainIndices, flush=True, file=fid_output_info)

# setting up minibatch gradient descent
num_miniBatch = int(trainSize / miniBatchSize)

# for inference using minibatch
remainder_test = len(testIndices) % miniBatchSize
if remainder_test > 0:
    testIndices += [dataIndices[i] for i in range(miniBatchSize-remainder_test)]
num_miniBatch_test = int(len(testIndices) / miniBatchSize)

# pre-computing for graph
grid_reference = tftools.get_reference_grid(size_target)
if sigma_filtering:
    k_smooth = tf.expand_dims(tf.expand_dims(tf.to_float(
        tfhelpers.get_smoothing_kernel(sigma_filtering/2**idx_size_ffd)), axis=3), axis=4)
sizes_target_hierarchy = tfhelpers.get_hierarchy_sizes(size_target, max([idx_size_ffd, len(lambda_bending), len(lambda_gradient)]))

transform_identity = tfhelpers.initial_transform_generator(miniBatchSize)
transform_initial = np.reshape(tfhelpers.initial_transform_generator(1), [-1, 12])  # single
feeder_target = tfhelpers.DataFeeder(h5fn_image_target, h5fn_label_target, h5fn_mask_target)
feeder_moving = tfhelpers.DataFeeder(h5fn_image_moving, h5fn_label_moving)

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
targetLabel0 = tftools.random_transform1(targetLabel_ph, targetTransform_ph, size_target) * (1-label_smoothing) + label_smoothing/2

strides_none = [1, 1, 1, 1, 1]
strides_down = [1, 2, 2, 2, 1]
k_conv = [3, 3, 3]
k_conv0 = [conv_size_initial, conv_size_initial, conv_size_initial]
k_pool = [1, 2, 2, 2, 1]


# --- global-net ---

if flag_use_global:
    # down-sampling
    nc0_g = num_channel_initial_global
    W0c_g = tf.get_variable("W0c_g", shape=k_conv0 + [2, nc0_g], initializer=tf.contrib.layers.xavier_initializer())
    W0r1_g = tf.get_variable("W0r1_g", shape=k_conv + [nc0_g, nc0_g], initializer=tf.contrib.layers.xavier_initializer())
    W0r2_g = tf.get_variable("W0r2_g", shape=k_conv + [nc0_g, nc0_g], initializer=tf.contrib.layers.xavier_initializer())
    vars_encode_g = [W0c_g, W0r1_g, W0r2_g]
    h0c_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(
        tf.concat([tftools.resize_volume(movingImage0, size_target), targetImage0], axis=4), W0c_g, strides_none,
        "SAME")))
    h0r1_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0c_g, W0r1_g, strides_none, "SAME")))
    h0r2_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0r1_g, W0r2_g, strides_none, "SAME")) + h0c_g)
    h0_g = tf.nn.max_pool3d(h0r2_g, k_pool, strides_down, padding="SAME")

    nc1_g = nc0_g * 2
    W1c_g = tf.get_variable("W1c_g", shape=k_conv + [nc0_g, nc1_g], initializer=tf.contrib.layers.xavier_initializer())
    W1r1_g = tf.get_variable("W1r1_g", shape=k_conv + [nc1_g, nc1_g], initializer=tf.contrib.layers.xavier_initializer())
    W1r2_g = tf.get_variable("W1r2_g", shape=k_conv + [nc1_g, nc1_g], initializer=tf.contrib.layers.xavier_initializer())
    vars_encode_g += [W1c_g, W1r1_g, W1r2_g]
    h1c_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0_g, W1c_g, strides_none, "SAME")))
    h1r1_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1c_g, W1r1_g, strides_none, "SAME")))
    h1r2_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1r1_g, W1r2_g, strides_none, "SAME")) + h1c_g)
    h1_g = tf.nn.max_pool3d(h1r2_g, k_pool, strides_down, padding="SAME")

    nc2_g = nc1_g * 2
    W2c_g = tf.get_variable("W2c_g", shape=k_conv + [nc1_g, nc2_g], initializer=tf.contrib.layers.xavier_initializer())
    W2r1_g = tf.get_variable("W2r1_g", shape=k_conv + [nc2_g, nc2_g], initializer=tf.contrib.layers.xavier_initializer())
    W2r2_g = tf.get_variable("W2r2_g", shape=k_conv + [nc2_g, nc2_g], initializer=tf.contrib.layers.xavier_initializer())
    vars_encode_g += [W2c_g, W2r1_g, W2r2_g]
    h2c_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1_g, W2c_g, strides_none, "SAME")))
    h2r1_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2c_g, W2r1_g, strides_none, "SAME")))
    h2r2_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2r1_g, W2r2_g, strides_none, "SAME")) + h2c_g)
    h2_g = tf.nn.max_pool3d(h2r2_g, k_pool, strides_down, padding="SAME")

    nc3_g = nc2_g * 2
    W3c_g = tf.get_variable("W3c_g", shape=k_conv + [nc2_g, nc3_g], initializer=tf.contrib.layers.xavier_initializer())
    W3r1_g = tf.get_variable("W3r1_g", shape=k_conv + [nc3_g, nc3_g], initializer=tf.contrib.layers.xavier_initializer())
    W3r2_g = tf.get_variable("W3r2_g", shape=k_conv + [nc3_g, nc3_g], initializer=tf.contrib.layers.xavier_initializer())
    vars_encode_g += [W3c_g, W3r1_g, W3r2_g]
    h3c_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2_g, W3c_g, strides_none, "SAME")))
    h3r1_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3c_g, W3r1_g, strides_none, "SAME")))
    h3r2_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3r1_g, W3r2_g, strides_none, "SAME")) + h3c_g)
    h3_g = tf.nn.max_pool3d(h3r2_g, k_pool, strides_down, padding="SAME")

    # deep
    ncD_g = nc3_g * 2  # deep layer
    WD_g = tf.get_variable("WD_g", shape=k_conv + [nc3_g, ncD_g], initializer=tf.contrib.layers.xavier_initializer())
    vars_encode_g += [WD_g]
    hD_g = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3_g, WD_g, strides_none, "SAME")))

    # - global resampler
    nfD = hD_g.shape.dims[1].value * hD_g.shape.dims[2].value * hD_g.shape.dims[3].value * hD_g.shape.dims[4].value
    W_global = tf.get_variable("W_global", shape=[nfD, 12], initializer=initialiser_global)
    b_global = tf.get_variable("b_global", shape=[1, 12], initializer=tf.constant_initializer(transform_initial+initial_bias))  # tf.Variable(tf.squeeze(tf.to_float(transform_identity+initial_bias), axis=1), name='b_global')
    vars_encode_g += [W_global]  # no bias term
    theta = tf.matmul(tf.reshape(hD_g, [miniBatchSize, -1]), W_global) + b_global
    grid_sample_global = tftools.warp_grid(grid_reference, theta)
    movingLabel_global = tftools.resample_linear(movingLabel0, grid_sample_global)
    displacement_global = grid_sample_global - grid_reference  # for saving/debug only


# --- local-net ---
if flag_use_local:
    # down-sampling
    nc0 = num_channel_initial
    W0c = tf.get_variable("W0c", shape=k_conv0+[2, nc0], initializer=tf.contrib.layers.xavier_initializer())
    W0r1 = tf.get_variable("W0r1", shape=k_conv+[nc0, nc0], initializer=tf.contrib.layers.xavier_initializer())
    W0r2 = tf.get_variable("W0r2", shape=k_conv+[nc0, nc0], initializer=tf.contrib.layers.xavier_initializer())
    vars_encode = [W0c, W0r1, W0r2]
    if flag_use_composite:
        h0c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(
            tf.concat([movingLabel_global, targetImage0], axis=4), W0c, strides_none, "SAME")))
    else:
        h0c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(
            tf.concat([tftools.resize_volume(movingImage0, size_target), targetImage0], axis=4), W0c, strides_none, "SAME")))
    h0r1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0c, W0r1, strides_none, "SAME")))
    h0r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0r1, W0r2, strides_none, "SAME")) + h0c)
    h0 = tf.nn.max_pool3d(h0r2, k_pool, strides_down, padding="SAME")

    nc1 = nc0*2
    W1c = tf.get_variable("W1c", shape=k_conv+[nc0, nc1], initializer=tf.contrib.layers.xavier_initializer())
    W1r1 = tf.get_variable("W1r1", shape=k_conv+[nc1, nc1], initializer=tf.contrib.layers.xavier_initializer())
    W1r2 = tf.get_variable("W1r2", shape=k_conv+[nc1, nc1], initializer=tf.contrib.layers.xavier_initializer())
    vars_encode += [W1c, W1r1, W1r2]
    h1c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0, W1c, strides_none, "SAME")))
    h1r1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1c, W1r1, strides_none, "SAME")))
    h1r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1r1, W1r2, strides_none, "SAME")) + h1c)
    h1 = tf.nn.max_pool3d(h1r2, k_pool, strides_down, padding="SAME")

    nc2 = nc1*2
    W2c = tf.get_variable("W2c", shape=k_conv+[nc1, nc2], initializer=tf.contrib.layers.xavier_initializer())
    W2r1 = tf.get_variable("W2r1", shape=k_conv+[nc2, nc2], initializer=tf.contrib.layers.xavier_initializer())
    W2r2 = tf.get_variable("W2r2", shape=k_conv+[nc2, nc2], initializer=tf.contrib.layers.xavier_initializer())
    vars_encode += [W2c, W2r1, W2r2]
    h2c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1, W2c, strides_none, "SAME")))
    h2r1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2c, W2r1, strides_none, "SAME")))
    h2r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2r1, W2r2, strides_none, "SAME")) + h2c)
    h2 = tf.nn.max_pool3d(h2r2, k_pool, strides_down, padding="SAME")

    nc3 = nc2*2
    W3c = tf.get_variable("W3c", shape=k_conv+[nc2, nc3], initializer=tf.contrib.layers.xavier_initializer())
    W3r1 = tf.get_variable("W3r1", shape=k_conv+[nc3, nc3], initializer=tf.contrib.layers.xavier_initializer())
    W3r2 = tf.get_variable("W3r2", shape=k_conv+[nc3, nc3], initializer=tf.contrib.layers.xavier_initializer())
    vars_encode += [W3c, W3r1, W3r2]
    h3c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2, W3c, strides_none, "SAME")))
    h3r1 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3c, W3r1, strides_none, "SAME")))
    h3r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3r1, W3r2, strides_none, "SAME")) + h3c)
    h3 = tf.nn.max_pool3d(h3r2, k_pool, strides_down, padding="SAME")

    # deep
    ncD = nc3*2  # deep layer
    WD = tf.get_variable("WD", shape=k_conv+[nc3, ncD], initializer=tf.contrib.layers.xavier_initializer())
    vars_encode += [WD]
    hD = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3, WD, strides_none, "SAME")))

    # up-sampling
    W3_c = tf.get_variable("W3_c", shape=k_conv+[nc3, ncD], initializer=tf.contrib.layers.xavier_initializer())
    W3_r1 = tf.get_variable("W3_r1", shape=k_conv+[nc3, nc3], initializer=tf.contrib.layers.xavier_initializer())
    W3_r2 = tf.get_variable("W3_r2", shape=k_conv+[nc3, nc3], initializer=tf.contrib.layers.xavier_initializer())
    vars_decode = [W3_c, W3_r1, W3_r2]
    h3_c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(hD, W3_c, h3c.get_shape(), strides_down, "SAME")))
    h3_r1 = tf.add(h3_c, h3c)
    h3_r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3_r1, W3_r1, strides_none, "SAME")))
    h3_ = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h3_r2, W3_r2, strides_none, "SAME")) + h3_r1)

    W2_c = tf.get_variable("W2_c", shape=k_conv+[nc2, nc3], initializer=tf.contrib.layers.xavier_initializer())
    W2_r1 = tf.get_variable("W2_r1", shape=k_conv+[nc2, nc2], initializer=tf.contrib.layers.xavier_initializer())
    W2_r2 = tf.get_variable("W2_r2", shape=k_conv+[nc2, nc2], initializer=tf.contrib.layers.xavier_initializer())
    vars_decode += [W2_c, W2_r1, W2_r2]
    h2_c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(h3_, W2_c, h2c.get_shape(), strides_down, "SAME")))
    h2_r1 = tf.add(h2_c, h2c)
    h2_r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2_r1, W2_r1, strides_none, "SAME")))
    h2_ = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h2_r2, W2_r2, strides_none, "SAME")) + h2_r1)

    W1_c = tf.get_variable("W1_c", shape=k_conv+[nc1, nc2], initializer=tf.contrib.layers.xavier_initializer())
    W1_r1 = tf.get_variable("W1_r1", shape=k_conv+[nc1, nc1], initializer=tf.contrib.layers.xavier_initializer())
    W1_r2 = tf.get_variable("W1_r2", shape=k_conv+[nc1, nc1], initializer=tf.contrib.layers.xavier_initializer())
    vars_decode += [W1_c, W1_r1, W1_r2]
    h1_c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(h2_, W1_c, h1c.get_shape(), strides_down, "SAME")))
    h1_r1 = tf.add(h1_c, h1c)
    h1_r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1_r1, W1_r1, strides_none, "SAME")))
    h1_ = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h1_r2, W1_r2, strides_none, "SAME")) + h1_r1)

    W0_c = tf.get_variable("W0_c", shape=k_conv+[nc0, nc1], initializer=tf.contrib.layers.xavier_initializer())
    W0_r1 = tf.get_variable("W0_r1", shape=k_conv+[nc0, nc0], initializer=tf.contrib.layers.xavier_initializer())
    W0_r2 = tf.get_variable("W0_r2", shape=k_conv+[nc0, nc0], initializer=tf.contrib.layers.xavier_initializer())
    vars_decode += [W0_c, W0_r1, W0_r2]
    h0_c = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d_transpose(h1_, W0_c, h0c.get_shape(), strides_down, "SAME")))
    h0_r1 = tf.add(h0_c, h0c)
    h0_r2 = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0_r1, W0_r1, strides_none, "SAME")))
    h0_ = tf.nn.relu(tf.contrib.layers.batch_norm(tf.nn.conv3d(h0_r2, W0_r2, strides_none, "SAME")) + h0_r1)

    # - local resampler
    W_local = tf.get_variable("W_local", shape=k_conv+[nc0, 3], initializer=initialiser_local)
    b_local = tf.get_variable("b_local", shape=[3], initializer=tf.constant_initializer(initial_bias))  # tf.Variable(tf.constant(initial_bias, shape=[3]), name='b_local')
    vars_decode += [W_local]  # no bias term
    displacement = tf.nn.conv3d(h0_, W_local, strides_none, "SAME") + b_local
    if sigma_filtering:
        if idx_size_ffd == 0:
            displacement = tftools.displacement_filtering(displacement, k_smooth)
        else:
            displacement = tftools.resize_volume(displacement, sizes_target_hierarchy[idx_size_ffd])
            displacement = tftools.displacement_filtering(displacement, k_smooth, size_target)
    # grid_sample_local = grid_reference + displacement  # for saving/debug only
    if flag_use_composite:
        movingLabel_composite = tftools.resample_linear(movingLabel0, grid_sample_global + displacement)
    else:
        movingLabel_local = tftools.resample_linear(movingLabel0, grid_reference + displacement)


# loss functions
if flag_use_global:
    tf.add_to_collection('loss_global', tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.concat([targetLabel0, 1-targetLabel0], axis=4),
        logits=tf.concat([movingLabel_global, 1-movingLabel_global], axis=4))))
if flag_use_local:
    if flag_use_composite:
        tf.add_to_collection('loss_composite', tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.concat([targetLabel0, 1-targetLabel0], axis=4),
            logits=tf.concat([movingLabel_composite, 1-movingLabel_composite], axis=4))))
    else:
        tf.add_to_collection('loss_local', tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.concat([targetLabel0, 1-targetLabel0], axis=4),
            logits=tf.concat([movingLabel_local, 1-movingLabel_local], axis=4))))


# regularisers
if flag_use_local:
    be = tftools.hierarchy_regulariser(displacement, lambda_bending, sizes_target_hierarchy, tftools.compute_bending_energy)
    gn = tftools.hierarchy_regulariser(displacement, lambda_gradient, sizes_target_hierarchy, tftools.compute_gradient_norm)
    tf.add_to_collection('loss_local', be)
    tf.add_to_collection('loss_local', gn)
    if flag_use_composite:
        tf.add_to_collection('loss_composite', be)
        tf.add_to_collection('loss_composite', gn)
else:
    be = tf.constant(-1.0)
    gn = tf.constant(-1.0)

# weight-decay
if lambda_decay > 0:
    if flag_use_global:
        for i in range(len(vars_encode_g)):
            tf.add_to_collection('loss_global', tf.nn.l2_loss(vars_encode_g[i]) * lambda_decay)
            if flag_use_composite: tf.add_to_collection('loss_composite', tf.nn.l2_loss(vars_encode_g[i]) * lambda_decay)
    if flag_use_local:
        if flag_use_composite:
            for i in range(len(vars_encode)):
                tf.add_to_collection('loss_composite', tf.nn.l2_loss(vars_encode[i]) * lambda_decay)
            for i in range(len(vars_decode)):
                tf.add_to_collection('loss_composite', tf.nn.l2_loss(vars_decode[i]) * lambda_decay)
        else:
            for i in range(len(vars_encode)):
                tf.add_to_collection('loss_local', tf.nn.l2_loss(vars_encode[i]) * lambda_decay)
            for i in range(len(vars_decode)):
                tf.add_to_collection('loss_local', tf.nn.l2_loss(vars_decode[i]) * lambda_decay)

# loss collections
if flag_use_global:
    loss_global = tf.add_n(tf.get_collection('loss_global'))
if flag_use_local:
    if flag_use_composite:
        loss_composite = tf.add_n(tf.get_collection('loss_composite'))
    else:
        loss_local = tf.add_n(tf.get_collection('loss_local'))


# utility nodes
# dice, movingVol, targetVol = tftools.compute_dice(movingLabel_warped, targetLabel0)
if flag_use_global:
    dice_global, vol1, vol2 = tftools.compute_dice(movingLabel_global, targetLabel0)
    dist_global = tftools.compute_centroid_distance(movingLabel_global, targetLabel0, grid_reference)
if flag_use_local:
    if flag_use_composite:
        dice_composite, vol1, vol2 = tftools.compute_dice(movingLabel_composite, targetLabel0)
        dist_composite = tftools.compute_centroid_distance(movingLabel_composite, targetLabel0, grid_reference)
    else:
        dice_local, vol1, vol2 = tftools.compute_dice(movingLabel_local, targetLabel0)
        dist_local = tftools.compute_centroid_distance(movingLabel_local, targetLabel0, grid_reference)


# setting up optimisation
if flag_use_global:
    # vars_global = vars_encode+[b_global]
    train_global = tf.train.AdamOptimizer(learning_rate).minimize(loss_global)
if flag_use_local:
    if flag_use_composite:
        # vars_composite = vars_encode+vars_decode+[b_global, b_local]
        train_composite = tf.train.AdamOptimizer(learning_rate).minimize(loss_composite)
    else:
        # vars_local = vars_encode+vars_decode+[b_local]
        train_local = tf.train.AdamOptimizer(learning_rate).minimize(loss_local)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# training
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

    if (not flag_single_label) & (step >= start_multiple_label):
        if random.random() < importance_sampling:
            label_indices = [random.randrange(feeder_moving.num_important[i]) for i in miniBatch_indices]  # hack - todo: use proper sampling
        else:
            label_indices = [random.randrange(feeder_moving.num_labels[i]) for i in miniBatch_indices]
    else:
        label_indices = [0] * miniBatchSize

    trainFeed = {movingImage_ph: feeder_moving.get_image_batch(miniBatch_indices),
                 targetImage_ph: feeder_target.get_image_batch(miniBatch_indices),
                 movingLabel_ph: feeder_moving.get_label_batch(miniBatch_indices, label_indices),
                 targetLabel_ph: feeder_target.get_label_batch(miniBatch_indices, label_indices),
                 movingTransform_ph: tfhelpers.random_transform_generator(miniBatchSize),
                 targetTransform_ph: tfhelpers.random_transform_generator(miniBatchSize)}

    if flag_use_composite & (step >= start_composite):
        sess.run(train_composite, feed_dict=trainFeed)
    elif flag_use_global:
        sess.run(train_global, feed_dict=trainFeed)
    elif flag_use_local:
        sess.run(train_local, feed_dict=trainFeed)

    if step in range(0, totalIterations, log_freq_info):
        if flag_use_composite & (step >= start_composite):
            loss_train, be_train, gn_train, dice_train, dist_train, dice_global_train, dist_global_train = sess.run(
                [loss_composite, be, gn, dice_composite, dist_composite, dice_global, dist_global], feed_dict=trainFeed)
            print('Optimiser: training composite transformation:')
            print('Optimiser: training composite transformation:', flush=True, file=fid_output_info)
            # --- debug ---
            # displacement_train = sess.run([displacement], feed_dict=trainFeed)
            # print('%s' % displacement_train)
        elif flag_use_global:
            loss_train, dice_train, dist_train = sess.run([loss_global, dice_global, dist_global], feed_dict=trainFeed)
            be_train, gn_train = -1, -1
            print('Optimiser: training global transformation:')
            print('Optimiser: training global transformation:', flush=True, file=fid_output_info)
            # --- debug ---
            # displacement_train = sess.run([displacement_global], feed_dict=trainFeed)
            # print('%s' % displacement_train)
        elif flag_use_local:
            loss_train, be_train, gn_train, dice_train, dist_train = sess.run([loss_local, be, gn, dice_local, dist_local], feed_dict=trainFeed)
            print('Optimiser: training local transformation:')
            print('Optimiser: training local transformation:', flush=True, file=fid_output_info)

        print('[%s] Step %d: loss=%f, be=%f, gn=%f' % (current_time, step, loss_train, be_train, gn_train))
        print('[%s] Step %d: loss=%f, be=%f, gn=%f' % (current_time, step, loss_train, be_train, gn_train), flush=True, file=fid_output_info)
        print('  Dice: %s' % dice_train)
        print('  Dice: %s' % dice_train, flush=True, file=fid_output_info)
        print('  Distance: %s' % dist_train)
        print('  Distance: %s' % dist_train, flush=True, file=fid_output_info)
        if flag_use_composite & (step >= start_composite):
            print('  Dice (global): %s' % dice_global_train)
            print('  Dice (global): %s' % dice_global_train, flush=True, file=fid_output_info)
            print('  Distance (global): %s' % dist_global_train)
            print('  Distance (global): %s' % dist_global_train, flush=True, file=fid_output_info)

        if flag_debugPrint:
            vol1_train, vol2_train = sess.run([vol1, vol2], feed_dict=trainFeed)
            print('DEBUG-PRINT: vol1: %s' % vol1_train)
            print('DEBUG-PRINT: vol1: %s' % vol1_train, flush=True, file=fid_output_info)
            print('DEBUG-PRINT: vol2: %s' % vol2_train)
            print('DEBUG-PRINT: vol2: %s' % vol2_train, flush=True, file=fid_output_info)
            print('DEBUG-PRINT: label_indices: %s' % label_indices)
            print('DEBUG-PRINT: label_indices: %s' % label_indices, flush=True, file=fid_output_info)

    # Debug data
    if step in range(log_start_debug, totalIterations, log_freq_debug):
        # save debug samples
        filename_log_data = "debug_data_i%09d.h5" % step
        fid_debug_data = h5py.File(os.path.join(dir_output, filename_log_data), 'w')

        fid_debug_data.create_dataset('/miniBatch_indices/', data=miniBatch_indices),
        fid_debug_data.create_dataset('/label_indices/', data=label_indices)

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
        if flag_use_composite & (step >= start_composite):
            displacement_global_train, displacement_train, movingLabel_warped_train = sess.run([displacement_global, displacement, movingLabel_composite], feed_dict=trainFeed)
            fid_debug_data.create_dataset('/displacement_train/', displacement_train.shape, dtype=displacement_train.dtype, data=displacement_train)
            fid_debug_data.create_dataset('/displacement_global_train/', displacement_global_train.shape, dtype=displacement_global_train.dtype, data=displacement_global_train)
        elif flag_use_global:
            displacement_global_train, movingLabel_warped_train = sess.run([displacement_global, movingLabel_global], feed_dict=trainFeed)
            fid_debug_data.create_dataset('/displacement_global_train/', displacement_global_train.shape, dtype=displacement_global_train.dtype, data=displacement_global_train)
        elif flag_use_local:
            displacement_train, movingLabel_warped_train = sess.run([displacement, movingLabel_local], feed_dict=trainFeed)
            fid_debug_data.create_dataset('/displacement_train/', displacement_train.shape, dtype=displacement_train.dtype, data=displacement_train)
        fid_debug_data.create_dataset('/movingLabel_warped_train/', movingLabel_warped_train.shape, dtype=movingLabel_warped_train.dtype, data=movingLabel_warped_train)
        # --------------------------

        # --- test ---
        fid_debug_data.create_dataset('/testIndices/', data=testIndices)
        for k in range(num_miniBatch_test):
            idx_test = [testIndices[i] for i in range(miniBatchSize*k, miniBatchSize*(k+1))]
            idx_label_test = [random.randrange(feeder_moving.num_important[i]) for i in idx_test]  # random test
            # idx_label_test = [0] * miniBatchSize  # use all the first ones for now
            testFeed = {movingImage_ph: feeder_moving.get_image_batch(idx_test),
                        targetImage_ph: feeder_target.get_image_batch(idx_test),
                        movingLabel_ph: feeder_moving.get_label_batch(idx_test, idx_label_test),
                        targetLabel_ph: feeder_target.get_label_batch(idx_test, idx_label_test),
                        movingTransform_ph: transform_identity,
                        targetTransform_ph: transform_identity}

            if flag_use_composite & (step >= start_composite):
                # test_t0 = time.time()
                displacement_global_test, displacement_test, movingLabel_warped_test = sess.run([displacement_global, displacement, movingLabel_composite], feed_dict=testFeed)
                # print('Elapsed time: %f second(s).' % (time.time() - test_t0), flush=True, file=fid_output_info)
                if k == (num_miniBatch_test - 1) & (remainder_test > 0):
                    movingLabel_warped_test = movingLabel_warped_test[:remainder_test, :]
                    displacement_global_test = displacement_global_test[:remainder_test, :]
                    displacement_test = displacement_test[:remainder_test, :]
                fid_debug_data.create_dataset('/movingLabel_warped_test_k%d/' % k, movingLabel_warped_test.shape, dtype=movingLabel_warped_test.dtype, data=movingLabel_warped_test)
                fid_debug_data.create_dataset('/displacement_test_k%d/' % k, displacement_test.shape, dtype=displacement_test.dtype, data=displacement_test)
                fid_debug_data.create_dataset('/displacement_global_test_k%d/' % k, displacement_global_test.shape, dtype=displacement_global_test.dtype, data=displacement_global_test)
            elif flag_use_global:
                displacement_global_test, movingLabel_warped_test = sess.run([displacement_global, movingLabel_global], feed_dict=testFeed)
                if k == (num_miniBatch_test - 1) & (remainder_test > 0):
                    movingLabel_warped_test = movingLabel_warped_test[:remainder_test, :]
                    displacement_global_test = displacement_global_test[:remainder_test, :]
                fid_debug_data.create_dataset('/movingLabel_warped_test_k%d/' % k, movingLabel_warped_test.shape, dtype=movingLabel_warped_test.dtype, data=movingLabel_warped_test)
                fid_debug_data.create_dataset('/displacement_global_test_k%d/' % k, displacement_global_test.shape, dtype=displacement_global_test.dtype, data=displacement_global_test)
            elif flag_use_local:
                displacement_test, movingLabel_warped_test = sess.run([displacement, movingLabel_local], feed_dict=testFeed)
                if k == (num_miniBatch_test - 1) & (remainder_test > 0):
                    movingLabel_warped_test = movingLabel_warped_test[:remainder_test, :]
                    displacement_test = displacement_test[:remainder_test, :]
                fid_debug_data.create_dataset('/movingLabel_warped_test_k%d/' % k, movingLabel_warped_test.shape, dtype=movingLabel_warped_test.dtype, data=movingLabel_warped_test)
                fid_debug_data.create_dataset('/displacement_test_k%d/' % k, displacement_test.shape, dtype=displacement_test.dtype, data=displacement_test)

            if flag_use_composite & (step >= start_composite):
                dice_test, dist_test = sess.run([dice_composite, dist_composite], feed_dict=testFeed)
            elif flag_use_global:
                dice_test, dist_test = sess.run([dice_global, dist_global], feed_dict=testFeed)
            elif flag_use_local:
                dice_test, dist_test = sess.run([dice_local, dist_local], feed_dict=testFeed)

            print('***test*** Dice: %s' % dice_test)
            print('***test*** Dice: %s' % dice_test, flush=True, file=fid_output_info)
            print('***test*** Distance: %s' % dist_test)
            print('***test*** Distance: %s' % dist_test, flush=True, file=fid_output_info)
            fid_debug_data.create_dataset('/dice_test_k%d/' % k, data=dice_test)
            fid_debug_data.create_dataset('/dist_test_k%d/' % k, data=dist_test)

            if flag_debugPrint:
                vol1_train, vol2_train = sess.run([vol1, vol2], feed_dict=trainFeed)
                print('DEBUG-PRINT: idx_label_test: %s' % idx_label_test)
                print('DEBUG-PRINT: idx_label_test: %s' % idx_label_test, flush=True, file=fid_output_info)
        # ------------

        # flush in the end
        fid_debug_data.flush()
        fid_debug_data.close()
        print('Debug data saved at Step %d' % step)
        print('Debug data saved at Step %d' % step, flush=True, file=fid_output_info)
        # NB. do not use continue here!
# ---------- End of Computational Graph ----------

fid_output_info.close()
