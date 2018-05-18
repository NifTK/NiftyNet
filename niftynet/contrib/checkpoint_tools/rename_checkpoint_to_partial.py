# -*- coding: utf-8 -*-
import sys,glob,csv

import tensorflow as tf

def rename_checkpoint_to_partial(source,target,transform):
  vars=tf.contrib.framework.list_variables(source)
  var_names = [v.split('/') for v,s in vars]
  transform_pairs = []
  for s_name,t_name in transform:
    print(vars,s_name,t_name)
    if s_name[-1]=='/' and t_name[-1]=='/': # scope
      s_names=s_name.split('/')[:-1]
      t_names=t_name.split('/')[:-1]
      transform_pairs += [('/'.join(v),'/'.join(t_names+v[len(s_names):])) for v in var_names if v[:len(s_names)]==s_names]
    elif s_name[-1]!='/' and t_name[-1]!='/': # variable
      if s_name in [v for v,s in vars]:
        transform_pairs.append((s_name,t_name))
    else:
      raise ValueError('Cannot rename a variable to a scope or vice versa: %s->%s' %(s_name,t_name))
  print(transform_pairs)
  g = tf.Graph()
  with g.as_default():
    with tf.Session() as sess:
      for s_name,t_name in transform_pairs:
        var = tf.contrib.framework.load_variable(source, s_name)
        var = tf.Variable(var, name=t_name)
      saver = tf.train.Saver()
      sess.run(tf.global_variables_initializer())
      saver.save(sess, target)
usage = \
"""%s source_checkpoint destination_checkpoint rename_file
rename_file has the format:

source_scope1,renamed_scope1
source_scope2/variable1,renamed_scope2/renamed_variable1

which will rename source_scope1/* to renamed_scope1/* and source_scope2/variable1 to renamed_scope2/renamed_variable1
""" %sys.argv[0]

def main(argv):
  if len(argv)<3:
    print(usage)
    return 2
  if not glob.glob(argv[0]+'.index'):
    print('Checkpoint %s does not exist' % argv[0])
    return 2
  if not glob.glob(argv[2]):
    print('Transform file %s does not exist' % argv[2])
    return 2
  with open(argv[2],'rb') as csvfile:
    r=csv.reader(csvfile)
    rows=[row for row in r]
  if any(len(row)!=2 for row in rows):
    print('Error %s: each line must have a source and target variable name' %(argv[2]))
    return 2
  rename_checkpoint_to_partial(argv[0],argv[1],argv[2])
  return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
