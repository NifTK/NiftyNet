from __future__ import absolute_import, print_function

import os
import unittest

import numpy as np
import tensorflow as tf

import niftynet.utilities.misc_csv as misc_csv
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer as HistNorm
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer as MVNorm
from niftynet.utilities.csv_table import CSVTable
from niftynet.utilities.filename_matching import KeywordsMatching


@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class HistTest(tf.test.TestCase):
    def test_volume_loader(self):
        expected_T1 = np.array([0.0, 8.24277910972, 21.4917343731,
                                27.0551695202, 32.6186046672, 43.5081573038,
                                53.3535675285, 61.9058849776, 70.0929786194,
                                73.9944243858, 77.7437509974, 88.5331971492,
                                100.0])
        expected_FLAIR = np.array([0.0, 5.36540863446, 15.5386130103,
                                   20.7431912042, 26.1536608309, 36.669150376,
                                   44.7821246138, 50.7930589961, 56.1703089214,
                                   59.2393548654, 63.1565641037, 78.7271261392,
                                   100.0])
        constraint_T1 = KeywordsMatching(['testing_data'], ['T1'],
                                         ['Parcellation'])
        constraint_FLAIR = KeywordsMatching(['testing_data'], ['FLAIR'], [])
        constraint_array = [constraint_FLAIR, constraint_T1]
        misc_csv.write_matched_filenames_to_csv(
            constraint_array, os.path.join('testing_data','TestPrepareInputHGG.csv'))
        csv_dict = {
            'input_image_file': os.path.join('testing_data','TestPrepareInputHGG.csv'),
            'target_image_file': None,
            'weight_map_file': None,
            'target_note': None}
        csv_loader = CSVTable(csv_dict=csv_dict,
                              modality_names=('FLAIR', 'T1'),
                              allow_missing=True)
        subject_list = csv_loader.to_subject_list()
        self.assertAllClose(len(subject_list), 4)

        model_file = os.path.join('testing_data','standardisation_models.txt')
        if os.path.exists(model_file):
            os.remove(model_file)
        masking_func = BinaryMaskingLayer(
            type='otsu_plus',
            multimod_fusion='or')
        hist_norm = HistNorm(
            models_filename=model_file,
            binary_masking_func=masking_func,
            norm_type='percentile',
            cutoff=(0.05, 0.95))
        hist_norm.train_normalisation_ref(subjects=subject_list)
        out_map = hist_norm.mapping
        self.assertAllClose(out_map['T1'], expected_T1)
        self.assertAllClose(out_map['FLAIR'], expected_FLAIR)

        # normalise a uniformly sampled random image
        test_shape = (20, 20, 20, 2, 3)
        rand_image = np.random.uniform(low=-10.0, high=10.0, size=test_shape)
        norm_image = np.copy(rand_image)
        norm_image, mask = hist_norm(norm_image)

        # apply mean std normalisation
        mv_norm = MVNorm(masking_func)
        norm_image, _ = mv_norm(norm_image, mask)

        # mapping should keep at least the order of the images
        rand_image = rand_image[:,:,:,1,2].flatten()
        norm_image = norm_image[:,:,:,1,2].flatten()
        order_before = rand_image[1:] > rand_image[:-1]
        order_after = norm_image[1:] > norm_image[:-1]
        self.assertAllClose(np.mean(norm_image), 0.0)
        self.assertAllClose(np.std(norm_image), 1.0)
        self.assertAllClose(order_before, order_after)


if __name__ == "__main__":
    tf.test.main()
