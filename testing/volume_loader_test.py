from __future__ import print_function, print_function, print_function, \
    print_function

import tensorflow as tf

from utilities.filename_matching import KeywordsMatching
import utilities.misc_csv as misc_csv
from layer.input_normalisation import HistogramNormalisationLayer as HistNorm
from layer.volume_loader import VolumeLoaderLayer
from utilities.csv_table import CSVTable


class SubjectTest(tf.test.TestCase):
    def test_volume_loader(self):
        constraint_T1 = KeywordsMatching(['./testing_data'], ['T1'],
                                            ['Parcellation'])
        constraint_FLAIR = KeywordsMatching(['./testing_data'], ['FLAIR'],
                                               [])
        constraint_array = [constraint_FLAIR, constraint_T1]
        misc_csv.write_matched_filenames_to_csv(
            constraint_array, './testing_data/TestPrepareInputHGG.csv')

        constraint_Label = KeywordsMatching(['./testing_data'],
                                               ['Parcellation'], [])
        misc_csv.write_matched_filenames_to_csv(
            [constraint_Label], './testing_data/TestPrepareOutputHGG.csv')

        csv_dict = {
            'input_image_file': './testing_data/TestPrepareInputHGG.csv',
            'target_image_file': './testing_data/TestPrepareOutputHGG.csv',
            'weight_map_file': None,
            'target_note': None}
        # 'target_note': './testing_data/TestComments.csv'}

        csv_loader = CSVTable(csv_dict=csv_dict,
                              modality_names=('FLAIR', 'T1'),
                              allow_missing=True)

        hist_norm = HistNorm(
            models_filename='./testing_data/standardisation_models.txt',
            multimod_mask_type='or',
            norm_type='percentile',
            mask_type='otsu_plus')

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
