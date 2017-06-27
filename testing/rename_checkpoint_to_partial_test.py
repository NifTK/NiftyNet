from __future__ import absolute_import, print_function

import tensorflow as tf
import tokenize,os,itertools
from utilities.rename_checkpoint_to_partial import rename_checkpoint_to_partial

class SubjectTest(tf.test.TestCase):
  def make_checkpoint(self,checkpoint_name,definition):
    scopes={}
    tf.reset_default_graph()
    [tf.Variable(definition[k],name=k) for k in definition]
    with tf.Session() as sess:
      saver=tf.train.Saver()
      sess.run(tf.global_variables_initializer())
      fn=os.path.join('testing_data',checkpoint_name)
      saver.save(sess,fn)
    return fn
  def load_checkpoint(self,checkpoint_name):
    vars=tf.contrib.framework.list_variables(checkpoint_name)
    return {v:tf.contrib.framework.load_variable(checkpoint_name, v) for v,s in vars}

  def generic(self,source,target,transform):
    source=self.make_checkpoint('chk1',source)
    target_file=os.path.join('testing_data','partial')
    rename_checkpoint_to_partial(source,target_file,transform)
    generated = self.load_checkpoint(target_file)
    self.assertItemsEqual(target.keys(),generated.keys())
    for k in target:
      self.assertAllClose(target[k],generated[k])
    
  def test_rename_vars(self):
    self.generic({'foo':[1],'bar/baz':[1,2,3],'bar/bing/boffin':[2]},
                 {'foo2':[1],'bar2/baz2':[1,2,3],'bar2/bing2/boffin2':[2]},
                 (('foo','foo2'),('bar/baz','bar2/baz2'),('bar/bing/boffin','bar2/bing2/boffin2')))
  def test_rename_scope(self):
    self.generic({'foo':[1],'bar/baz':[1,2,3],'bar/bing/boffin':[2]},
                 {'foo2':[1],'bar2/baz':[1,2,3],'bar2/bing/boffin':[2]},
                 (('foo','foo2'),('bar/','bar2/')))
                 
if __name__ == "__main__":
    tf.test.main()
