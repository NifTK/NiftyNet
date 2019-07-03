# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

from niftynet.engine.sampler_random_vector_v2 import RandomVectorSampler
from tests.niftynet_testcase import NiftyNetTestCase

class RandomVectorSamplerTest(NiftyNetTestCase):
    def test_random_vector(self):
        sampler = RandomVectorSampler(names=('test_vector',),
                                      vector_size=(100,),
                                      batch_size=20,
                                      repeat=None)
        with self.cached_session() as sess:
            sampler.set_num_threads(2)
            out = sess.run(sampler.pop_batch_op())
            self.assertAllClose(out['test_vector'].shape, (20, 100))
        sampler.close_all()

    def test_ill_init(self):
        with self.assertRaisesRegexp(TypeError, ""):
            sampler = RandomVectorSampler(names=('test_vector',),
                                          vector_size=10,
                                          batch_size=20)
        with self.assertRaisesRegexp(TypeError, ""):
            sampler = RandomVectorSampler(names=0,
                                          vector_size=(10,),
                                          batch_size=20)

    def test_repeat(self):
        batch_size = 20
        n_interpolations = 10
        repeat = 4
        sampler = RandomVectorSampler(names=('test_vector',),
                                      vector_size=(100,),
                                      batch_size=batch_size,
                                      n_interpolations=n_interpolations,
                                      repeat=repeat)
        with self.cached_session() as sess:
            sampler.set_num_threads(1)
            n_output = 0
            for _ in range(2):
                out_vector = sess.run(sampler.pop_batch_op())
                if np.all(out_vector['test_vector'] == -1):
                    break
                n_output = n_output + batch_size
                self.assertAllClose(out_vector['test_vector'].shape,
                                    (batch_size, 100))
                self.assertAllClose(np.mean(out_vector['test_vector']),
                                    0.0, atol=0.5, rtol=0.5)
                self.assertAllClose(np.std(out_vector['test_vector']),
                                    1.0, atol=1.0, rtol=1.0)
            self.assertEquals(n_output, n_interpolations * repeat)
        sampler.close_all()


if __name__ == "__main__":
    tf.test.main()
