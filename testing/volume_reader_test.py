import tensorflow as tf

import utilities.constraints_classes as cc
import utilities.misc_csv as misc_csv
import utilities.parse_user_params as parse_user_params
import utilities.volume_reader as vr
from nn.preprocess import HistNormaliser_bis
from utilities.csv_table import CSVTable


class SubjectTest(tf.test.TestCase):
    def test_volume_reader(self):
        param = parse_user_params.run()
        constraint_T1 = cc.ConstraintSearch(
                ['./testing_data'], ['T1'], ['Parcellation'], ['_'])
        constraint_FLAIR = cc.ConstraintSearch(
                ['./testing_data'], ['FLAIR'], [], ['_'])
        constraint_array = [constraint_FLAIR, constraint_T1]
        misc_csv.write_matched_filenames_to_csv(
                constraint_array, './testing_data/TestPrepareInputHGG.csv')

        constraint_Label = cc.ConstraintSearch(
                ['./testing_data'], ['Parcellation'], [], ['_'])
        misc_csv.write_matched_filenames_to_csv(
                [constraint_Label], './testing_data/TestPrepareOutputHGG.csv')

        csv_dict = {'input_image_file': './testing_data/TestPrepareInputHGG.csv',
                    'target_image_file': './testing_data/TestPrepareOutputHGG.csv',
                    'weight_map_file': None,
                    'target_note': None}
        # 'target_note': './testing_data/TestComments.csv'}

        csv_loader = CSVTable(csv_dict=csv_dict,
                              modality_names=('FLAIR', 'T1'),
                              allow_missing=True)

        histogram_normalisor = HistNormaliser_bis(
            models_filename=param.saving_norm_dir,
            multimod_mask_type='or',
            norm_type=param.norm_type,
            cutoff=[x for x in param.norm_cutoff],
            mask_type='otsu_plus')

        new_vr = vr.VolumePreprocessor(csv_loader,
                                       histogram_normalisor,
                                       do_reorientation=True,
                                       do_resampling=True,
                                       do_normalisation=True,
                                       output_columns=(0, 1, 2),
                                       interp_order=(3, 0))

        img, seg, weight_map, subject = new_vr.next_subject()
        print img.shape
        if seg is not None:
            print seg.shape
        print weight_map
        print subject
        img, seg, weight_map, subject = new_vr.next_subject()


if __name__ == "__main__":
    tf.test.main()
