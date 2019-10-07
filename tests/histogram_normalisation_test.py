# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os

import numpy as np
import tensorflow as tf

from niftynet.io.image_reader import ImageReader
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.utilities.util_common import ParserNamespace
from tests.niftynet_testcase import NiftyNetTestCase

DATA_PARAM = {
    'T1': ParserNamespace(
        csv_file=os.path.join('testing_data', 'T1.csv'),
        path_to_search='testing_data',
        filename_contains=('_o_T1_time',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        loader=None
    ),
    'FLAIR': ParserNamespace(
        csv_file=os.path.join('testing_data', 'FLAIR.csv'),
        path_to_search='testing_data',
        filename_contains=('FLAIR_',),
        filename_not_contains=('Parcellation',),
        interp_order=3,
        pixdim=None,
        axcodes=None,
        loader=None
    )
}
TASK_PARAM = ParserNamespace(image=('T1', 'FLAIR'))
MODEL_FILE = os.path.join('testing_data', 'std_models.txt')
data_partitioner = ImageSetsPartitioner()
file_list = data_partitioner.initialise(DATA_PARAM).get_file_list()


# @unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class HistTest(NiftyNetTestCase):
    def test_volume_loader(self):
        expected_T1 = np.array(
            [0.0, 8.24277910972, 21.4917343731,
             27.0551695202, 32.6186046672, 43.5081573038,
             53.3535675285, 61.9058849776, 70.0929786194,
             73.9944243858, 77.7437509974, 88.5331971492,
             100.0])
        expected_FLAIR = np.array(
            [0.0, 5.36540863446, 15.5386130103,
             20.7431912042, 26.1536608309, 36.669150376,
             44.7821246138, 50.7930589961, 56.1703089214,
             59.2393548654, 63.1565641037, 78.7271261392,
             100.0])

        reader = ImageReader(['image'])
        reader.initialise(DATA_PARAM, TASK_PARAM, file_list)
        self.assertAllClose(len(reader._file_list), 4)

        foreground_masking_layer = BinaryMaskingLayer(
            type_str='otsu_plus',
            multimod_fusion='or')
        hist_norm = HistogramNormalisationLayer(
            image_name='image',
            modalities=vars(TASK_PARAM).get('image'),
            model_filename=MODEL_FILE,
            binary_masking_func=foreground_masking_layer,
            cutoff=(0.05, 0.95),
            name='hist_norm_layer')
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)
        hist_norm.train(reader.output_list)
        out_map = hist_norm.mapping

        self.assertAllClose(out_map['T1'], expected_T1)
        self.assertAllClose(out_map['FLAIR'], expected_FLAIR)

        # normalise a uniformly sampled random image
        test_shape = (20, 20, 20, 3, 2)
        rand_image = np.random.uniform(low=-10.0, high=10.0, size=test_shape)
        norm_image = np.copy(rand_image)
        norm_image_dict, mask_dict = hist_norm({'image': norm_image})
        norm_image, mask = hist_norm(norm_image, mask_dict)
        self.assertAllClose(norm_image_dict['image'], norm_image)
        self.assertAllClose(mask_dict['image'], mask)

        # apply mean std normalisation
        mv_norm = MeanVarNormalisationLayer(
            image_name='image',
            binary_masking_func=foreground_masking_layer)
        norm_image, _ = mv_norm(norm_image, mask)
        self.assertAllClose(norm_image.shape, mask.shape)

        mv_norm = MeanVarNormalisationLayer(
            image_name='image',
            binary_masking_func=None)
        norm_image, _ = mv_norm(norm_image)

        # mapping should keep at least the order of the images
        rand_image = rand_image[:, :, :, 1, 1].flatten()
        norm_image = norm_image[:, :, :, 1, 1].flatten()

        order_before = rand_image[1:] > rand_image[:-1]
        order_after = norm_image[1:] > norm_image[:-1]
        self.assertAllClose(np.mean(norm_image), 0.0)
        self.assertAllClose(np.std(norm_image), 1.0)
        self.assertAllClose(order_before, order_after)
        if os.path.exists(MODEL_FILE):
            os.remove(MODEL_FILE)


if __name__ == "__main__":
    tf.test.main()
