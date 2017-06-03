import tensorflow as tf

# sampler
from engine.selective_sampler import SelectiveSampler
from engine.spatial_location_check import SpatialLocationCheckLayer
from engine.volume_loader import VolumeLoaderLayer
from layer.input_normalisation import HistogramNormalisationLayer as HistNorm
from utilities.csv_table import CSVTable
from utilities.input_placeholders import ImagePatch


class SubjectTest(tf.test.TestCase):
    def test_volume_reader(self):

        csv_dict = {'input_image_file': './testing_data/testing_case_input',
                    'target_image_file': './testing_data/testing_case_target',
                    'weight_map_file': None,
                    'target_note': None}
        csv_loader = CSVTable(csv_dict=csv_dict,
                              modality_names=('FLAIR', 'T1'),
                              allow_missing=True)

        hist_norm = HistNorm(
            models_filename='./testing_data/standardisation_models.txt',
            multimod_mask_type='or',
            norm_type='percentile',
            mask_type='otsu_plus')

        volume_loader = VolumeLoaderLayer(csv_loader,
                                          is_training=True)
        print('found {} subjects'.format(len(volume_loader.subject_list)))

        # define output element patch
        patch_holder = ImagePatch(image_size=32,
                                  label_size=32,
                                  weight_map_size=None,
                                  spatial_rank=2,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_image_modality=2,
                                  num_label_modality=1,
                                  num_weight_map=0)
        location_check_layer = SpatialLocationCheckLayer(
            compulsory=[[0], [0.01]],
            minimum_ratio=0.01,
            min_numb_labels=1,
            padding=0)

        sampler = SelectiveSampler(
            patch=patch_holder,
            volume_loader=volume_loader,
            spatial_location_check=location_check_layer,
            data_augmentation_methods=[],
            patch_per_volume=10)

        n_volumes = 0
        for d in sampler():
            assert isinstance(d, ImagePatch)
            data_dict = d.as_dict(sampler.placeholders)
            self.assertAllClose((32, 32, 2), d.image.shape)
            self.assertAllClose((5,), d.info.shape)
            self.assertAllClose((32, 32, 1), d.label.shape)
            print(d.info)
            n_volumes = n_volumes + 1
            if n_volumes == 5:
                break

            keys = data_dict.keys()[0]
            output = data_dict.values()[0]
            for (idx, key) in enumerate(keys):
                print(key, output[idx].shape)


if __name__ == "__main__":
    tf.test.main()
