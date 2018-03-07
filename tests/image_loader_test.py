from __future__ import absolute_import, print_function

from os import listdir
from os.path import isfile, join
import tensorflow as tf
from niftynet.layer.crop import CropLayer
import niftynet.io.image_loader as image_loader

CASE_NIBABEL_3D = 'testing_data/FLAIR_1023.nii.gz'
CASE_LOGO_2D = 'niftynet-logo.png'

class ImageLoaderTest(tf.test.TestCase):
    def test_nibabel_3d(self):
        data = image_loader.load_image_from_file(CASE_NIBABEL_3D).get_data()
        self.assertAllClose(data.shape, (256, 168, 256))

    def load_2d_image(self, loader=None):
        data = image_loader.load_image_from_file(CASE_LOGO_2D, loader=loader).get_data()
        self.assertAllClose(data.shape, (400, 677, 1, 1, 4))

    def test_2d_loaders(self):
        with self.assertRaisesRegexp(ValueError, ''):
            self.load_2d_image('test')
        self.load_2d_image()
        for _loader in image_loader.AVAILABLE_LOADERS.keys():
            print('testing {}'.format(_loader))
            if _loader == 'nibabel':
                continue # skip nibabel for 2d image
            if _loader == 'fake':
                continue # skip the toy example
            self.load_2d_image(_loader)

    def test_all_data(self):
        folder = 'testing_data'
        all_files = [join(folder, f)
            for f in listdir(folder) if isfile(join(folder, f))]

        for f in all_files:
            if f.endswith(('nii.gz')):
                loaded_shape = image_loader.load_image_from_file(f).get_data().shape
                print(loaded_shape)
                self.assertGreaterEqual(5, len(loaded_shape))
            else:
                with self.assertRaisesRegexp(ValueError, ''):
                    image_loader.load_image_from_file(f)


if __name__ == "__main__":
    tf.test.main()
