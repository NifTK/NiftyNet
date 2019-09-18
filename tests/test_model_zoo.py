# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, unittest
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

from niftynet.utilities.download import download
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig
from niftynet import main as niftynet_main
from niftynet.application.base_application import SingletonApplication
from tests.niftynet_testcase import NiftyNetTestCase

MODEL_HOME = NiftyNetGlobalConfig().get_niftynet_home_folder()

def net_run_with_sys_argv(argv):
    # for gift-adelie
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    SingletonApplication.clear()
    cache = sys.argv
    argv.extend(['--cuda_devices', '0'])
    sys.argv = argv
    niftynet_main()
    sys.argv = cache


@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class DenseVNetAbdominalCTModelZooTest(NiftyNetTestCase):
    zoo_id = 'dense_vnet_abdominal_ct_model_zoo'
    location = 'dense_vnet_abdominal_ct'
    config = os.path.join(MODEL_HOME, 'extensions', 'dense_vnet_abdominal_ct', 'config.ini')
    application = 'net_segment'
    expected_output = os.path.join('segmentation_output','window_seg_100___niftynet_out.nii.gz')

    def setUp(self):
        NiftyNetTestCase.setUp(self)
        download(self.zoo_id, download_if_already_existing=True, verbose=False)

    def test_train_infer(self):
        self._train()
        self._infer()

    def _train(self):
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.config, 'train', '--max_iter', '2'])
        checkpoint = os.path.join(MODEL_HOME, 'models', self.location, 'models', 'model.ckpt-2.index')
        self.assertTrue(os.path.exists(checkpoint), 'Expected {} to exist.'.format(checkpoint))

    def _infer(self):
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.config, 'inference'])
        output = os.path.join(MODEL_HOME, 'models', self.location, self.expected_output)
        self.assertTrue(os.path.exists(output), 'Expected {} to exist.'.format(output))


@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class UltrasoundSimulatorGanModelZooTest(NiftyNetTestCase):
    zoo_id = 'ultrasound_simulator_gan_model_zoo'
    location = 'ultrasound_simulator_gan'
    config = os.path.join(MODEL_HOME, 'extensions', 'ultrasound_simulator_gan', 'config.ini')
    application = 'net_gan'
    expected_output = os.path.join('ultrasound_gan_simulated','5_000053__window_image_niftynet_generated.nii.gz')

    def setUp(self):
        NiftyNetTestCase.setUp(self)
        download(self.zoo_id, download_if_already_existing=True, verbose=False)

    def test_inference(self):
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.config, 'inference'])
        output = os.path.join(MODEL_HOME, 'models', self.location, self.expected_output)
        self.assertTrue(os.path.exists(output), 'Expected {} to exist.'.format(output))


@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class Highres3dnetBrainParcellationModelZooTest(NiftyNetTestCase):
    zoo_id = 'highres3dnet_brain_parcellation_model_zoo'
    location = 'highres3dnet_brain_parcellation'
    config = os.path.join(MODEL_HOME, 'extensions', 'highres3dnet_brain_parcellation', 'highres3dnet_config_eval.ini')
    application = 'net_segment'
    expected_output = os.path.join('parcellation_output','window_seg_OAS1_0145_MR2_mpr_n4_anon_sbj_111__niftynet_out.nii.gz')

    def setUp(self):
        NiftyNetTestCase.setUp(self)
        download(self.zoo_id, download_if_already_existing=True, verbose=False)

    def test_inference(self):
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.config, 'inference'])
        output = os.path.join(MODEL_HOME, 'models', self.location, self.expected_output)
        self.assertTrue(os.path.exists(output), 'Expected {} to exist.'.format(output))


@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class AnisotropicNetsBratsChallengeModelZooTest(NiftyNetTestCase):
    zoo_id = 'anisotropic_nets_brats_challenge_model_zoo'
    location = 'anisotropic_nets_brats_challenge'
    application = 'anisotropic_nets_brats_challenge.brats_seg_app.BRATSApp'
    expected_outputs = [os.path.join('model_whole_tumor_axial','pred_whole_tumor_axial','window_LGG71__niftynet_out.nii.gz'),
                        os.path.join('model_whole_tumor_coronal','pred_whole_tumor_coronal','window_LGG71__niftynet_out.nii.gz'),
                        os.path.join('model_whole_tumor_sagittal','pred_whole_tumor_sagittal','window_LGG71__niftynet_out.nii.gz')]
    configA = os.path.join(MODEL_HOME, 'extensions', 'anisotropic_nets_brats_challenge', 'whole_tumor_axial.ini')
    configC = os.path.join(MODEL_HOME, 'extensions', 'anisotropic_nets_brats_challenge', 'whole_tumor_coronal.ini')
    configS = os.path.join(MODEL_HOME, 'extensions', 'anisotropic_nets_brats_challenge', 'whole_tumor_sagittal.ini')

    def setUp(self):
        NiftyNetTestCase.setUp(self)
        download(self.zoo_id, download_if_already_existing=True, verbose=False)

    def test_inference(self):
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.configA, 'inference'])
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.configC, 'inference'])
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.configS, 'inference'])
        for eo in self.expected_outputs:
            output = os.path.join(MODEL_HOME, 'models', self.location, eo)
            self.assertTrue(os.path.exists(output), 'Expected {} to exist.'.format(output))


@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class MRCTRegressionModelZooTest(NiftyNetTestCase):
    zoo_id = 'mr_ct_regression_model_zoo'
    location = 'mr_ct_regression'
    application = 'niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression'
    config = os.path.join(MODEL_HOME, 'extensions', 'mr_ct_regression','net_isampler.ini')
    expected_output_train = [
        os.path.join('error_maps','CRI.nii.gz'),
        ]

    def setUp(self):
        NiftyNetTestCase.setUp(self)
        download(self.zoo_id, download_if_already_existing=True, verbose=False)

    def test_train(self):
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.config, 'train', '--starting_iter','0','--max_iter', '2'])
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.config, 'inference', '--inference_iter','2','--error_map','True'])
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.config, 'train', '--starting_iter','2','--max_iter','4'])

        checkpoint = os.path.join(MODEL_HOME, 'models', self.location, 'models', 'model.ckpt-2.index')
        self.assertTrue(os.path.exists(checkpoint), 'Expected {} to exist.'.format(checkpoint))
        checkpoint = os.path.join(MODEL_HOME, 'models', self.location, 'models', 'model.ckpt-4.index')
        self.assertTrue(os.path.exists(checkpoint), 'Expected {} to exist.'.format(checkpoint))
        for eo in self.expected_output_train:
            output = os.path.join(MODEL_HOME, 'models', self.location, eo)
            self.assertTrue(os.path.exists(output), 'Expected {} to exist.'.format(output))


@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class AutoContextMRCTModelZooTest(NiftyNetTestCase):
    zoo_id = 'autocontext_mr_ct_model_zoo'
    location = 'autocontext_mr_ct'
    application = 'net_regress'
    config = os.path.join(MODEL_HOME, 'extensions', 'autocontext_mr_ct','net_autocontext.ini')
    expected_output_train = [
        os.path.join('error_maps','WEB.nii.gz'),
        ]
    expected_output_inference = [
        os.path.join('autocontext_output','window_reg_CHA_niftynet_out.nii.gz'),
        ]


    def setUp(self):
        NiftyNetTestCase.setUp(self)
        download(self.zoo_id, download_if_already_existing=True, verbose=False)

    def test_train_infer(self):
        self._train()
        self._infer()

    def _train(self):
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.config, 'train', '--starting_iter','0','--max_iter', '2'])
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.config, 'inference', '--inference_iter','2'])
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.config, 'train', '--starting_iter','2','--max_iter','4'])

        checkpoint = os.path.join(MODEL_HOME, 'models', self.location, 'models', 'model.ckpt-2.index')
        self.assertTrue(os.path.exists(checkpoint))
        checkpoint = os.path.join(MODEL_HOME, 'models', self.location, 'models', 'model.ckpt-4.index')
        self.assertTrue(os.path.exists(checkpoint))
        for eo in self.expected_output_train:
            output = os.path.join(MODEL_HOME, 'models', self.location, eo)
            self.assertTrue(os.path.exists(output), 'Expected {} to exist.'.format(output))

    def _infer(self):
        net_run_with_sys_argv(['net_run', '-a', self.application, '-c', self.config, 'inference', '--inference_iter','-1'])
        for eo in self.expected_output_inference:
            output = os.path.join(MODEL_HOME, 'models', self.location, eo)
            self.assertTrue(os.path.exists(output), 'Expected {} to exist.'.format(output))


if __name__ == "__main__":
    tf.test.main()
