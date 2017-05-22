import tensorflow as tf
# for the preprocessor
from utilities.volume_reader import VolumePreprocessor
import utilities.volume_reader as vr
import utilities.parse_user_params as parse_user_params

# sampler
from layer.uniform_sampler import UniformSampler
from layer.input_placeholders import ImagePatch

class SubjectTest(tf.test.TestCase):

    def test_volume_reader(self):
        param = parse_user_params.run()
        dict_constraint = vr.Constraints([],[],[],[],[],[],[],False,False)
        dict_constraint._update_dict_constraint(param)
        dict_normalisation = vr.Normalisation('','')
        dict_normalisation._update_dict_normalisation(param)
        dict_masking = vr.Masking()
        dict_masking._update_dict_masking(param)

        # initialise volume loader
        volume_loader = VolumePreprocessor(
            dict_constraint=dict_constraint,
            dict_normalisation=dict_normalisation,
            dict_masking=dict_masking)
        print('found {} subjects'.format(len(volume_loader.subject_list)))
        for x in volume_loader.subject_list:
            print(x.file_path_dict.values())
        #out = volume_loader.next_subject()

        # define output element patch
        patch_holder = ImagePatch(image_shape=(32, 32, 32),
                                  label_shape=(32, 32, 32),
                                  weight_map_shape=None,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_image_modality=2,
                                  num_label_modality=2,
                                  num_map=1)

        sampler = UniformSampler(patch=patch_holder,
                                 volume_loader=volume_loader,
                                 name='uniform_sampler')
        for d in sampler():
            assert isinstance(d, ImagePatch)
            data_dict = d.as_dict()
            self.assertAllClose((32, 32, 32, 2), d.image.shape)
            self.assertAllClose((7,), d.info.shape)
            self.assertAllClose((32, 32, 32, 2), d.label.shape)
            print(d.info)

            keys = data_dict.keys()[0]
            output = data_dict.values()[0]
            for (idx, key) in enumerate(keys):
                print(key, output[idx].shape)

if __name__ == "__main__":
    tf.test.main()
