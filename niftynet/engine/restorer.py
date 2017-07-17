import tensorflow as tf
import os
from niftynet.utilities.restore_initializer import restore_initializer
from tensorflow.contrib.framework import list_variables
RESTORABLE='NiftyNetObjectsToRestore'

def resolve_checkpoint(checkpoint_name):
    # For now only supports checkpoint_name where
    # checkpoint_name.index is in the file system
    # eventually will support checkpoint names that can be referenced
    # in a paths file
    if os.path.exists(checkpoint_name+'.index'):
        return checkpoint_name
    raise ValueError('Invalid checkpoint {}'.format(checkpoint_name))
def global_variables_initialize_or_restorer(var_list=None):
    # For any scope added to RESTORABLE collection:
    # variable will be restored from a checkpoint if it exists in the
    # specified checkpoint and no scope ancestor can restore it.
    if var_list is None:
        var_list = tf.global_variables()
    restorable = sorted(tf.get_collection(RESTORABLE),key=lambda x: x[0])
    restored_vars={}
    for scope, checkpoint_name, checkpoint_scope in restorable:
        variables_in_scope = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope=scope)
        checkpoint_file=resolve_checkpoint(checkpoint_name)
        variables_in_file = [v for v,s in list_variables(checkpoint_file)]
        rename = lambda x: x.replace(scope,checkpoint_scope).replace(':0','')
        to_restore = [v for v in variables_in_scope if v in var_list and \
                                                       rename(v.name) in variables_in_file]
        for var in to_restore:
            if var in restored_vars:
                continue
            checkpoint_subscope,var_name = rename(var.name).rsplit('/',1)
            initializer = restore_initializer(checkpoint_name,var_name,checkpoint_subscope)
            restored_vars[var] = tf.assign(var, initializer(var.get_shape(),dtype=var.dtype))
    init_others=tf.variables_initializer([v for v in var_list if v not in restored_vars])
    restore_op = tf.group(init_others,*list(restored_vars.values()))
    return restore_op


