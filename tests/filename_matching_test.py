from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.utilities.filename_matching import KeywordsMatching
from tests.niftynet_testcase import NiftyNetTestCase

class FileNameMatchingTest(NiftyNetTestCase):
    def test_default(self):
        matcher = KeywordsMatching()
        with self.assertRaisesRegexp(ValueError, ""):
            matcher.matching_subjects_and_filenames()
        with self.assertRaisesRegexp(AttributeError, ""):
            KeywordsMatching.from_dict('wrong_argument')

    def test_from_dict(self):
        with self.assertRaisesRegexp(ValueError, ""):
            KeywordsMatching.from_dict({'path_to_search': 'wrong_folder'})
        matcher = KeywordsMatching.from_dict(
            {'path_to_search': 'testing_data/images2d'})
        f_list, s_list = matcher.matching_subjects_and_filenames()
        self.assertEqual(len(f_list), 30)
        self.assertEqual(len(s_list), 30)
        self.assertEqual(s_list[0][0], 'img0_g')

    def test_keywords_grep(self):
        matcher = KeywordsMatching.from_dict(
            {'path_to_search': 'testing_data/images2d',
             'filename_contains': 'img'})
        f_list, s_list = matcher.matching_subjects_and_filenames()
        self.assertEqual(len(f_list), 30)
        self.assertEqual(len(s_list), 30)
        # filename matched 'img' will return and
        # the matched string is removed from subject_id
        self.assertEqual(s_list[0][0], '0_g')

    def test_keywords_not_contain(self):
        matcher = KeywordsMatching.from_dict(
            {'path_to_search': 'testing_data/images2d',
             'filename_not_contains': 'img'})
        with self.assertRaisesRegexp(ValueError, ""):
            # not filename (not containing 'img') matched
            matcher.matching_subjects_and_filenames()

        matcher = KeywordsMatching.from_dict(
            {'path_to_search': 'testing_data/images2d',
             'filename_not_contains': ('_m', '_u')})
        f_list, s_list = matcher.matching_subjects_and_filenames()
        self.assertEqual(len(f_list), 10)
        self.assertEqual(len(s_list), 10)

        matcher_comp = KeywordsMatching.from_dict(
            {'path_to_search': 'testing_data/images2d',
             'filename_contains': '_g'})
        f_comp, s_comp = matcher_comp.matching_subjects_and_filenames()
        self.assertEqual(len(f_comp), 10)
        self.assertEqual(len(f_comp), 10)
        self.assertEqual(f_comp, f_list)

    def test_keywords_params_combine(self):
        matcher = KeywordsMatching.from_dict(
            {'path_to_search': 'testing_data/images2d',
             'filename_contains': '_g',
             'filename_removefromid': 'img|_g'})
        f_list, s_list = matcher.matching_subjects_and_filenames()
        self.assertEqual(len(f_list), 10)
        self.assertEqual(len(s_list), 10)

        matcher_comp = KeywordsMatching.from_dict(
            {'path_to_search': 'testing_data/images2d',
             'filename_not_contains': ('_m', '_u'),
             'filename_removefromid': "img|_g"})
        f_comp, s_comp = matcher_comp.matching_subjects_and_filenames()
        self.assertEqual(f_comp, f_list)
        self.assertEqual(s_comp, s_list)

        matcher = KeywordsMatching.from_dict(
            {'path_to_search': 'testing_data/images2d',
             'filename_removefromid': 'img|_g|_m|_u'})
        with self.assertRaisesRegexp(ValueError, ""):
            matcher.matching_subjects_and_filenames()


if __name__ == "__main__":
    tf.test.main()
