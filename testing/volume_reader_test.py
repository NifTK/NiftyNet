import tensorflow as tf
from utilities.volume_reader import VolumePreprocessor
import utilities.volume_reader as vr
import utilities.parse_user_params as parse_user_params
import utilities.constraints_classes as cc
import utilities.misc_csv as misc_csv

class SubjectTest(tf.test.TestCase):

    def test_volume_reader(self):
        param = parse_user_params.run()
        constraint_T1 = cc.ConstraintSearch(['./testing_data'], ['T1'], ['Parcellation'], ['_'])
        constraint_FLAIR = cc.ConstraintSearch(['./testing_data'], ['FLAIR'], [], ['_'])
        constraint_array = [constraint_FLAIR, constraint_T1]
        misc_csv.create_csv_prepare5d(constraint_array, './testing_data/TestPrepareInputHGG.csv')

        constraint_Label = cc.ConstraintSearch(['./testing_data'], ['Parcellation'], [], ['_'])
        misc_csv.create_csv_prepare5d([constraint_Label], './testing_data/TestPrepareOutputHGG.csv')

        csv_list = cc.InputList('./testing_data/TestPrepareInputHGG.csv',
                                './testing_data/TestPrepareOutputHGG.csv',
                                None, None, None)
        #csv_list = cc.InputList('./testing_data/TestPrepareInputHGG.csv',
        #                        None, None, None, None)

        list_modalities = {'T1': 1, 'FLAIR': 0}

        dict_normalisation = cc.Normalisation('', '')
        dict_normalisation._update_dict_normalisation(param)
        dict_masking = cc.Masking()
        dict_masking._update_dict_masking(param)
        dict_normalisation.list_modalities = list_modalities
        new_vr = vr.VolumePreprocessor(dict_normalisation, dict_masking, csv_list=csv_list,
                                       number_list=cc.InputList(4, 1, None, None, None),
                                       loss=['dice'],
                                       flags=cc.Flags(flag_reorient=True, flag_resample=True,
                                                      flag_standardise=True))

        img, seg, weight_map, subject = new_vr.next_subject()
        print img.shape
        if seg is not None:
            print seg.shape
        print weight_map
        print subject



if __name__ == "__main__":
    tf.test.main()




