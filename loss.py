import numpy as np
import tensorflow as tf


class LossFunction(object):
    def __init__(self, n_class, loss_type='dice', reg_type='l2', decay=0.0):
        self.num_classes = n_class
        self.set_loss_type(loss_type)
        self.set_reg_type(reg_type)
        self.set_decay(decay)
        print('Training loss: {}_loss + ({})*{}_loss'.format(
            loss_type, decay, reg_type))

    def set_loss_type(self, type_str):
        if type_str == "cross_entropy":
            self.data_loss_fun = cross_entropy
        elif type_str == "dice":
            self.data_loss_fun = dice
        elif type_str == 'GDSC':
            self.data_loss_fun = GDSC_loss
        elif type_str == 'SensSpec':
            self.data_loss_fun = SensSpec_loss

    def set_reg_type(self, type_str):
        if type_str == "l2":
            self.reg_loss_fun = l2_reg_loss

    def set_decay(self, decay):
        self.decay = decay

    def total_loss(self, pred, labels, var_scope):
        with tf.device('/cpu:0'):
            # data term
            pred = tf.reshape(pred, [-1, self.num_classes])
            labels = tf.reshape(labels, [-1])
            data_loss = self.data_loss_fun(pred, labels)
            if self.decay <= 0:
                return data_loss

            # regularisation term
            reg_loss = self.reg_loss_fun(var_scope)
            return tf.add(data_loss, self.decay * reg_loss, name='total_loss')

# Generalised Dice score with different type weights
def GDSC_loss(pred,labels,type_weight='Square'):
    n_voxels = labels.get_shape()[0].value
    n_classes = pred.get_shape()[1].value
    ids = tf.constant(np.array(range(n_voxels)), dtype=tf.int64)
    ids = tf.stack([ids, labels], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=[1.0] * n_voxels,
                              dense_shape=[n_voxels, n_classes])
    ref_vol = tf.sparse_reduce_sum(one_hot,reduction_axes=[0])+0.1
    intersect = tf.sparse_reduce_sum(one_hot * pred , reduction_axes=[0])
    seg_vol = tf.reduce_sum(pred,0)+0.1
    if type_weight=='Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight=='Simple':
        weights = tf.reciprocal(ref_vol)
    else:
        weights = tf.one_like(ref_vol)
    GDSC = tf.reduce_sum(tf.multiply(weights,intersect))/tf.reduce_sum(tf.multiply(weights,seg_vol+ref_vol))
    return 1-GDSC

# Sensitivity Specificity loss function adapted to work for multiple labels
def SensSpec_loss(pred,labels,r=0.05):
    n_voxels = labels.get_shape()[0].value
    n_classes = pred.get_shape()[1].value
    ids = tf.constant(np.array(range(n_voxels)), dtype=tf.int64)
    ids = tf.stack([ids, labels], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=[1.0] * n_voxels,
                              dense_shape=[n_voxels, n_classes])
    one_hotB = 1 - tf.sparse_tensor_to_dense(one_hot)
    SensSpec = tf.reduce_mean(tf.add(tf.multiply(r , tf.reduce_sum(tf.multiply(tf.square(-1*tf.sparse_add(-1*pred , one_hot)) \
               , tf.sparse_tensor_to_dense(one_hot)), 0) / tf.sparse_reduce_sum(one_hot, 0)),\
                tf.multiply((1 - r) , tf.reduce_sum(tf.multiply(tf.square(-1*tf.sparse_add(-1*pred, one_hot)), \
                one_hotB), 0) / tf.reduce_sum(one_hotB, 0))))
    return SensSpec

def l2_reg_loss(scope):
    if tf.get_collection('reg_var', scope) == []:
        return 0.0
    return tf.add_n([tf.nn.l2_loss(reg_var) for reg_var in
                     tf.get_collection('reg_var', scope)])


def cross_entropy(pred, labels):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(pred, labels)
    return tf.reduce_mean(entropy)


def dice(pred, labels):
    n_voxels = labels.get_shape()[0].value
    n_classes = pred.get_shape()[1].value
    pred = tf.nn.softmax(pred)
    # construct sparse matrix for labels to save space
    ids = tf.constant(np.array(range(n_voxels)), dtype=tf.int64)
    ids = tf.stack([ids, labels], axis=1)
    one_hot = tf.SparseTensor(indices=ids,
                              values=[1.0] * n_voxels,
                              dense_shape=[n_voxels, n_classes])
    # dice
    score = (2 * tf.sparse_reduce_sum(one_hot * pred, reduction_axes=[0])) / \
            (tf.reduce_sum(tf.square(pred), reduction_indices=[0]) + \
             tf.sparse_reduce_sum(one_hot, reduction_axes=[0]) + \
             0.00001)
    score.set_shape([n_classes])
    # minimising average 1 - dice_coefficients
    return 1.0 - tf.reduce_mean(score)
