# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, unittest

import tensorflow as tf

from niftynet.utilities.download import download
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig
from niftynet import main as niftynet_main
from niftynet.application.base_application import SingletonApplication
    
MODEL_HOME = NiftyNetGlobalConfig().get_niftynet_home_folder()

def net_run_with_sys_argv(argv):
    cache = sys.argv
    sys.argv = argv
    niftynet_main()
    sys.argv = cache
    
class ModelZooTestMixin(object):
    def test_train(self):
        self.assertTrue(download('dense_vnet_abdominal_ct_model_zoo', True))
        SingletonApplication.clear()
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', 
            os.path.join(MODEL_HOME, 'extensions', self.location, self.config), 
            'train', '--max_iter', '2'])
        checkpoint = os.path.join(MODEL_HOME, 'models', self.location, 'models', 'model.ckpt-1.index')
        self.assertTrue(os.path.exists(checkpoint))
        self.check_train()    
        
    def test_inference(self):
        self.assertTrue(download('dense_vnet_abdominal_ct_model_zoo', True))
        SingletonApplication.clear()
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', 
            os.path.join(MODEL_HOME, 'extensions', self.location, self.config), 
            'inference'])
        output = os.path.join(MODEL_HOME, 'models', self.location, self.expected_output)
        self.assertTrue(os.path.exists(output))
        self.check_inference()    
        
    def check_inference(self):
        pass
        
    def check_train(self):
        pass
        
        
@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class DenseVNetAbdominalCTModelZooTest(tf.test.TestCase, ModelZooTestMixin):
    id = 'dense_vnet_abdominal_ct_model_zoo'
    location = 'dense_vnet_abdominal_ct'
    config = 'config.ini'
    application = 'net_segment'
    expected_output = os.path.join('segmentation_output','100__niftynet_out.nii.gz')

if __name__ == "__main__":
    tf.test.main()
