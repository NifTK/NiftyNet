from __future__ import absolute_import, print_function

import tensorflow as tf
import os
import niftynet.utilities.misc_csv as misc_csv
from niftynet.engine.volume_loader import VolumeLoaderLayer
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer as HistNorm
from niftynet.utilities.csv_table import CSVTable
from niftynet.utilities.filename_matching import KeywordsMatching


class SubjectTest(tf.test.TestCase):
    def test_volume_loader(self):
        constraint_T1 = KeywordsMatching(['testing_data'], ['T1'],
                                         ['Parcellation'])
        constraint_FLAIR = KeywordsMatching(['testing_data'], ['FLAIR'],
                                            [])
        constraint_array = [constraint_FLAIR, constraint_T1]
        misc_csv.write_matched_filenames_to_csv(
            constraint_array, os.path.join('testing_data','TestPrepareInputHGG.csv'))

        constraint_Label = KeywordsMatching(['testing_data'],
                                            ['Parcellation'], [])
        misc_csv.write_matched_filenames_to_csv(
            [constraint_Label], os.path.join('testing_data','TestPrepareOutputHGG.csv'))

        csv_dict = {
            'input_image_file': os.path.join('testing_data','TestPrepareInputHGG.csv'),
            'target_image_file': os.path.join('testing_data','TestPrepareOutputHGG.csv'),
            'weight_map_file': None,
            'target_note': None}
        # 'target_note': os.path.join('testing_data','TestComments.csv')}

        csv_loader = CSVTable(csv_dict=csv_dict,
                              modality_names=('FLAIR', 'T1'),
                              allow_missing=True)

        hist_norm = HistNorm(
            models_filename=os.path.join('testing_data','standardisation_models.txt'),
            binary_masking_func=BinaryMaskingLayer(
                type='otsu_plus',
                multimod_fusion='or'),
            norm_type='percentile',
            cutoff=(0.01, 0.99))

        volume_loader = VolumeLoaderLayer(csv_loader, hist_norm)

        img, seg, weight_map, subject = volume_loader()
        print(img.data.shape)
        if seg is not None:
            print(seg.data.shape)
        print(weight_map)
        print(volume_loader.subject_list[subject])
        img, seg, weight_map, subject = volume_loader()
        print(subject)


if __name__ == "__main__":
    tf.test.main()
