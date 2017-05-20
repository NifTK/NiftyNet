import tensorflow as tf
from utilities.volume_reader import VolumePreprocessor
import utilities.volume_reader as vr
import utilities.parse_user_params as parse_user_params
import utilities.misc_grep_file as misc

class SubjectTest(tf.test.TestCase):

    def test_volume_reader(self):
        param = parse_user_params.run()
        dict_constraint = vr.Constraints([],[],[],[],[],[],[],False,False)
        dict_constraint._update_dict_constraint(param)
        dict_normalisation = vr.Normalisation('','')
        dict_normalisation._update_dict_normalisation(param)
        dict_masking = vr.Masking()
        dict_masking._update_dict_masking(param)

        a_processor = VolumePreprocessor(
            dict_constraint=dict_constraint,
            dict_normalisation=dict_normalisation,
            dict_masking=dict_masking)
        print a_processor.subject_list
        out = _processor.next_subject()

if __name__ == "__main__":
    tf.test.main()
