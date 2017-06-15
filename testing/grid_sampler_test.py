from __future__ import absolute_import, print_function

import tensorflow as tf

# sampler
from engine.grid_sampler import GridSampler
from engine.volume_loader import VolumeLoaderLayer
from layer.binary_masking import BinaryMaskingLayer
from layer.histogram_normalisation import \
    HistogramNormalisationLayer as HistNorm
from layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer as MVNorm
from utilities.csv_table import CSVTable
from utilities.input_placeholders import ImagePatch


class UniformSamplerTest(tf.test.TestCase):
    def test_3d(self):
        csv_dict = {'input_image_file': './testing_data/testing_case_input',
                    'target_image_file': './testing_data/testing_case_target',
                    'weight_map_file': None,
                    'target_note': None}
        csv_loader = CSVTable(csv_dict=csv_dict,
                              modality_names=('FLAIR', 'T1'),
                              allow_missing=True)

        masking_func = BinaryMaskingLayer(
            type='otsu_plus',
            multimod_fusion='or')
        hist_norm = HistNorm(
            models_filename='./testing_data/standardisation_models.txt',
            binary_masking_func=masking_func,
            norm_type='percentile',
            cutoff=(0.01, 0.99))
        mv_norm = MVNorm(binary_masking_func=masking_func)
        volume_loader = VolumeLoaderLayer(csv_loader,
                                          standardisor=(hist_norm, mv_norm),
                                          is_training=False)
        print('found {} subjects'.format(len(volume_loader.subject_list)))

        # define output element patch
        patch_holder = ImagePatch(image_size=32,
                                  label_size=32,
                                  weight_map_size=None,
                                  spatial_rank=3,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_image_modality=2,
                                  num_label_modality=1,
                                  num_weight_map=0)

        # generate samples and probably return to the original space
        sampler = GridSampler(patch=patch_holder,
                              volume_loader=volume_loader,
                              grid_size=100,
                              name='grid_sampler')
        n_volumes = 0
        for d in sampler():
            assert isinstance(d, ImagePatch)
            data_dict = d.as_dict(sampler.placeholders)
            self.assertAllClose((32, 32, 32, 2), d.image.shape)
            self.assertAllClose((7,), d.info.shape)
            self.assertAllClose((32, 32, 32, 1), d.label.shape)
            n_volumes = n_volumes + 1
            if n_volumes == 3:
                break
                # if n_volumes == 5:
                #    break

                #    keys = data_dict.keys()[0]
                #    output = data_dict.values()[0]
                #    for (idx, key) in enumerate(keys):
                #        print(key, output[idx].shape)

    def test_2d(self):
        csv_dict = {'input_image_file': './testing_data/testing_case_input',
                    'target_image_file': './testing_data/testing_case_target',
                    'weight_map_file': None,
                    'target_note': None}
        csv_loader = CSVTable(csv_dict=csv_dict,
                              modality_names=('FLAIR', 'T1'),
                              allow_missing=True)

        masking_func = BinaryMaskingLayer(
            type='otsu_plus',
            multimod_fusion='or')
        hist_norm = HistNorm(
            models_filename='./testing_data/standardisation_models.txt',
            binary_masking_func=masking_func,
            norm_type='percentile',
            cutoff=(0.01, 0.99))
        mv_norm = MVNorm(binary_masking_func=masking_func)

        volume_loader = VolumeLoaderLayer(csv_loader,
                                          (hist_norm, mv_norm),
                                          is_training=False)
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

        # generate samples and probably return to the original space
        sampler = GridSampler(patch=patch_holder,
                              volume_loader=volume_loader,
                              grid_size=100,
                              name='grid_sampler')
        n_volumes = 0
        for d in sampler():
            assert isinstance(d, ImagePatch)
            data_dict = d.as_dict(sampler.placeholders)
            self.assertAllClose((32, 32, 2), d.image.shape)
            self.assertAllClose((5,), d.info.shape)
            self.assertAllClose((32, 32, 1), d.label.shape)
            n_volumes = n_volumes + 1
            if n_volumes == 3:
                break

    def test_25d(self):
        csv_dict = {'input_image_file': './testing_data/testing_case_input',
                    'target_image_file': './testing_data/testing_case_target',
                    'weight_map_file': None,
                    'target_note': None}
        csv_loader = CSVTable(csv_dict=csv_dict,
                              modality_names=('FLAIR', 'T1'),
                              allow_missing=True)

        masking_func = BinaryMaskingLayer(
            type='otsu_plus',
            multimod_fusion='or')
        hist_norm = HistNorm(
            models_filename='./testing_data/standardisation_models.txt',
            binary_masking_func=masking_func,
            norm_type='percentile',
            cutoff=(0.01, 0.99))
        mv_norm = MVNorm(binary_masking_func=masking_func)

        volume_loader = VolumeLoaderLayer(csv_loader,
                                          (hist_norm, mv_norm),
                                          is_training=False)
        print('found {} subjects'.format(len(volume_loader.subject_list)))

        # define output element patch
        patch_holder = ImagePatch(image_size=32,
                                  label_size=32,
                                  weight_map_size=None,
                                  spatial_rank=2.5,
                                  image_dtype=tf.float32,
                                  label_dtype=tf.int64,
                                  num_image_modality=2,
                                  num_label_modality=1,
                                  num_weight_map=0)

        # generate samples and probably return to the original space
        sampler = GridSampler(patch=patch_holder,
                              volume_loader=volume_loader,
                              grid_size=100,
                              name='grid_sampler')
        n_volumes = 0
        for d in sampler():
            assert isinstance(d, ImagePatch)
            data_dict = d.as_dict(sampler.placeholders)
            self.assertAllClose((32, 32, 2), d.image.shape)
            self.assertAllClose((6,), d.info.shape)
            self.assertAllClose((32, 32, 1), d.label.shape)
            n_volumes = n_volumes + 1
            if n_volumes == 3:
                break


if __name__ == "__main__":
    tf.test.main()
