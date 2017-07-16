from __future__ import absolute_import, print_function

import tensorflow as tf
import numpy as np

import tokenize, os, itertools
import niftynet.engine.restorer

from niftynet.layer.convolution import ConvolutionalLayer

class RestorerTest(tf.test.TestCase):
    def make_checkpoint(self, checkpoint_name, definition):
        scopes = {}
        tf.reset_default_graph()
        [tf.Variable(definition[k], name=k,dtype=np.float32) for k in definition]
        with tf.Session() as sess:
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            fn = os.path.join('testing_data', checkpoint_name)
            saver.save(sess, fn)
        return fn

    def test_restore_block(self):
        definition= {'foo': [1], 'bar/conv_/w': np.random.randn(3,3,1,3),
                     'bar2/conv_/w': np.random.randn(3,3,1,3),
                     'foo3/conv_/w': np.random.randn(3,3,1,3),
                     'bar/bing/boffin': [2]}
        checkpoint_name = self.make_checkpoint('chk1',definition)
        tf.reset_default_graph()
        block1 = ConvolutionalLayer(3,3,with_bn=False,name='foo')
        b1=block1(tf.ones([1.,5.,5.,1.]))
        tf.add_to_collection(niftynet.engine.restorer.RESTORABLE,('foo',checkpoint_name,'bar'))
        block2 = ConvolutionalLayer(4,3,name='bar',with_bn=False,
                                    w_initializer=tf.constant_initializer(1.))
        b2=block2(tf.ones([1., 5., 5., 1.]))
        block3 = ConvolutionalLayer(3, 3, with_bn=False, name='foo2')
        block3.restore_from_checkpoint(checkpoint_name,'bar2')
        b3=block3(tf.ones([1.,5.,5.,1.]))
        block4 = ConvolutionalLayer(3, 3, with_bn=False, name='foo3')
        block4.restore_from_checkpoint(checkpoint_name)
        b4=block4(tf.ones([1.,5.,5.,1.]))
        tf.add_to_collection(niftynet.engine.restorer.RESTORABLE,('foo',checkpoint_name,'bar'))
        init_op = niftynet.engine.restorer.global_variables_initialize_or_restorer()
        all_vars = tf.global_variables()
        with self.test_session() as sess:
            sess.run(init_op)
            foo_w_var = [v for v in all_vars if v.name == block1.layer_scope().name+'/conv_/w:0'][0]
            bar_w_var = [v for v in all_vars if v.name == block2.layer_scope().name+'/conv_/w:0'][0]
            foo2_w_var = [v for v in all_vars if v.name == block3.layer_scope().name + '/conv_/w:0'][0]
            foo3_w_var = [v for v in all_vars if v.name == block4.layer_scope().name + '/conv_/w:0'][0]
            [foo_w,bar_w,foo2_w,foo3_w]=sess.run([foo_w_var,bar_w_var,foo2_w_var,foo3_w_var])
            self.assertAllClose(foo_w, definition['bar/conv_/w'])
            self.assertAllClose(bar_w,np.ones([3,3,1,4]))
            self.assertAllClose(foo2_w, definition['bar2/conv_/w'])
            self.assertAllClose(foo3_w, definition['foo3/conv_/w'])

    def test_no_restores(self):
        tf.reset_default_graph()
        block1 = ConvolutionalLayer(4,3,name='bar',with_bn=False,
                                    w_initializer=tf.constant_initializer(1.))
        b2=block1(tf.ones([1., 5., 5., 1.]))
        init_op = niftynet.engine.restorer.global_variables_initialize_or_restorer()
        all_vars = tf.global_variables()
        with self.test_session() as sess:
            sess.run(init_op)
            bar_w_var = [v for v in all_vars if v.name == block1.layer_scope().name+'/conv_/w:0'][0]
            [bar_w]=sess.run([bar_w_var])
            self.assertAllClose(bar_w,np.ones([3,3,1,4]))


if __name__ == "__main__":
    tf.test.main()
