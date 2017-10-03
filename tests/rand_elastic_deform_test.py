from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.contrib.layer.rand_elastic_deform import RandomElasticDeformationLayer

#Better test on some sample images with some groundtruth than just try to run
#the code
class RandElasticDeformationTest(tf.test.TestCase):
    def get_4d_input(self):
        input_4d = {'testdata': np.ones((32, 32, 32, 8))}
        interp_order = {'testdata': (3,) * 8}
        return input_4d, interp_order

    def get_5d_input(self):
        input_5d = {'testdata': np.ones((64, 64, 64, 8, 1))}
        interp_order = {'testdata': (3,)}
        return input_5d, interp_order

    def test_4d_shape(self):
        x, interp_orders = self.get_4d_input()
        self.deform(interp_orders, x)

    def test_5d_shape(self):
        x, interp_orders = self.get_5d_input()
        self.deform(interp_orders, x)

    def deform(self, interp_orders, x):
        rand_deformation_layer = RandomElasticDeformationLayer(
            shapes={'testdata_shape': x.values()[0].shape}, num_controlpoints=4, std_deformation_sigma=15)
        rand_deformation_layer.randomise()
        out = rand_deformation_layer(x, interp_orders)

def run_test():
    try:
        from niftynet.contrib.layer.rand_elastic_deform import RandomElasticDeformationLayer
    except ImportError:
        print("SimpleITK not found, skip elastic deformation test")
        return
    tf.test.main()

if __name__ == "__main__":
    run_test()