# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

from niftynet.engine.image_window import ImageWindow
from niftynet.utilities.util_common import ParserNamespace
from tests.niftynet_testcase import NiftyNetTestCase


def get_static_window_param():
    return dict(
        source_names={
            'image': (u'modality1', u'modality2'),
            'label': (u'modality3',)},
        image_shapes={
            'image': (192, 160, 192, 1, 2),
            'label': (192, 160, 192, 1, 1)},
        image_dtypes={
            'image': tf.float32,
            'label': tf.float32},
        window_sizes={
            'modality1': ParserNamespace(spatial_window_size=(10, 10, 2)),
            'modality2': (10, 10, 2),
            'modality3': (5, 5, 1)}
    )


def get_dynamic_image_window():
    return dict(
        source_names={
            'image': (u'modality1', u'modality2'),
            'label': (u'modality3',)},
        image_shapes={
            'image': (192, 160, 192, 1, 2),
            'label': (192, 160, 192, 1, 1)},
        image_dtypes={
            'image': tf.float32,
            'label': tf.float32},
        window_sizes={
            'modality1': ParserNamespace(spatial_window_size=(10, 10)),
            'modality2': ParserNamespace(spatial_window_size=(10, 10)),
            'modality3': ParserNamespace(spatial_window_size=(2,))},
        allow_dynamic=True
    )


def get_all_dynamic_image_window():
    return dict(
        source_names={
            'image': (u'modality1', u'modality2'),
            'label': (u'modality3',)},
        image_shapes={
            'image': (192, 160, 192, 1, 2),
            'label': (192, 160, 192, 1, 1)},
        image_dtypes={
            'image': tf.float32,
            'label': tf.float32},
        window_sizes={
            'modality1': ParserNamespace(spatial_window_size=(-1, -1)),
            'modality2': ParserNamespace(spatial_window_size=(-1, -1)),
            'modality3': ParserNamespace(spatial_window_size=(-1,))},
        allow_dynamic=True
    )


def get_ill_image_window():
    return dict(
        source_names={
            'image': (u'modality1',),
            'label': (u'modality3',)},
        image_shapes={
            'image': (192, 160, 192, 1, 2),
            'label': (192, 160, 192, 1, 1)},
        image_dtypes={
            'image': tf.float32,
            'label': tf.float32},
        window_sizes={
            'modality1': ParserNamespace(spatial_window_size=(10, 10)),
            'modality3': ParserNamespace(spatial_window_size=())}
    )


def get_ill_image_window_1():
    return dict(
        source_names={
            'image': (u'modality1', u'modality2'),
            'label': (u'modality3',)},
        image_shapes={
            'image': (192, 160, 192, 1, 2),
            'label': (192, 160, 192, 1, 1)},
        image_dtypes={
            'image': tf.float32,
            'label': tf.float32},
        window_sizes={
            'modality1': ParserNamespace(spatial_window_size=(10, 10)),
            'modality2': ParserNamespace(spatial_window_size=(10, 10)),
            'modality3': ParserNamespace(spatial_window_size=())}
    )


def get_ill_image_window_2():
    return dict(
        source_names={
            'image': (u'modality1', u'modality2'),
            'label': (u'modality3',)},
        image_shapes={
            'image': (192, 160, 192, 1, 2),
            'label': (192, 160, 192, 1, 1)},
        image_dtypes={
            'image': tf.float32},
        window_sizes={
            'modality1': ParserNamespace(spatial_window_size=(10, 10)),
            'modality2': ParserNamespace(spatial_window_size=(10, 10)),
            'modality3': ParserNamespace(spatial_window_size=())}
    )


# def get_ill_image_window_3():
#    return dict(
#        source_names={
#            'image': (u'modality1', u'modality2'),
#            'label': (u'modality3',)},
#        image_shapes={
#            'image': (192, 160, 192, 1, 2),
#            'label': (192, 160, 192, 1, 1)},
#        image_dtypes={
#            'image': tf.float32},
#        data_param={
#            'modality1': ParserNamespace(spatial_window_size=(10, 10)),
#            'modality2': ParserNamespace(spatial_window_size=(10, 10)),
#            'modality3': ParserNamespace(spatial_window_size=())}
#
#    )


class ImageWindowTest(NiftyNetTestCase):
    def test_init(self):
        window = ImageWindow.from_data_reader_properties(
            **get_static_window_param())
        self.assertAllEqual(
            window.placeholders_dict(1)['image'].shape.as_list(),
            [1, 10, 10, 2, 1, 2])
        self.assertAllEqual(
            window.placeholders_dict(1)['label'].shape.as_list(),
            [1, 5, 5, 1, 1, 1])

        window = ImageWindow.from_data_reader_properties(
            **get_dynamic_image_window())
        self.assertAllEqual(
            window.placeholders_dict(1)['image'].shape.as_list(),
            [1, 10, 10, None, 1, 2])
        self.assertAllEqual(
            window.placeholders_dict(1)['label'].shape.as_list(),
            [1, 2, None, None, 1, 1])

        window = ImageWindow.from_data_reader_properties(
            **get_all_dynamic_image_window())
        self.assertAllEqual(
            window.placeholders_dict(1)['image'].shape.as_list(),
            [1, None, None, None, 1, 2])
        self.assertAllEqual(
            window.placeholders_dict(1)['label'].shape.as_list(),
            [1, None, None, None, 1, 1])

    def test_ill_cases(self):
        with self.assertRaisesRegexp(ValueError, ""):
            ImageWindow.from_data_reader_properties(
                **get_ill_image_window())

        with self.assertRaisesRegexp(ValueError, ""):
            ImageWindow.from_data_reader_properties(
                **get_ill_image_window_1())

        with self.assertRaisesRegexp(ValueError, ""):
            ImageWindow.from_data_reader_properties(
                **get_ill_image_window_2())

    def test_matching_image_shapes(self):
        to_match = {
            'image': (42, 43, 44, 1, 2),
            'label': (42, 43, 44, 1, 2)}

        new_shape = ImageWindow.from_data_reader_properties(
            **get_static_window_param()).match_image_shapes(to_match)
        self.assertAllEqual(new_shape['image'], [10, 10, 2, 1, 2])
        self.assertAllEqual(new_shape['label'], [5, 5, 1, 1, 1])

        new_shape = ImageWindow.from_data_reader_properties(
            **get_dynamic_image_window()).match_image_shapes(to_match)
        self.assertAllEqual(new_shape['image'], [10, 10, 44, 1, 2])
        self.assertAllEqual(new_shape['label'], [2, 43, 44, 1, 1])

        new_shape = ImageWindow.from_data_reader_properties(
            **get_all_dynamic_image_window()).match_image_shapes(to_match)
        self.assertAllEqual(new_shape['image'], [42, 43, 44, 1, 2])
        self.assertAllEqual(new_shape['label'], [42, 43, 44, 1, 1])

    def test_placeholders(self):
        window = ImageWindow.from_data_reader_properties(
            **get_static_window_param())
        window.placeholders_dict(10)

        self.assertAllEqual(
            window.image_data_placeholder('image').shape.as_list(),
            [10, 10, 10, 2, 1, 2])
        self.assertAllEqual(
            window.image_data_placeholder('image').dtype,
            window.tf_dtypes['image'])
        self.assertAllEqual(
            window.coordinates_placeholder('image').shape.as_list(),
            [10, 7])
        self.assertAllEqual(
            window.coordinates_placeholder('image').dtype, tf.int32)

        self.assertAllEqual(
            window.image_data_placeholder('label').shape.as_list(),
            [10, 5, 5, 1, 1, 1])
        self.assertAllEqual(
            window.image_data_placeholder('label').dtype,
            window.tf_dtypes['label'])
        self.assertAllEqual(
            window.coordinates_placeholder('label').shape.as_list(),
            [10, 7])
        self.assertAllEqual(
            window.coordinates_placeholder('label').dtype, tf.int32)

        window = ImageWindow.from_data_reader_properties(
            **get_dynamic_image_window())
        window.placeholders_dict(10)

        self.assertAllEqual(
            window.image_data_placeholder('image').shape.as_list(),
            [1, 10, 10, None, 1, 2])
        self.assertAllEqual(
            window.image_data_placeholder('image').dtype,
            window.tf_dtypes['image'])
        self.assertAllEqual(
            window.coordinates_placeholder('image').shape.as_list(),
            [1, 7])
        self.assertAllEqual(
            window.coordinates_placeholder('image').dtype, tf.int32)

        self.assertAllEqual(
            window.image_data_placeholder('label').shape.as_list(),
            [1, 2, None, None, 1, 1])
        self.assertAllEqual(
            window.image_data_placeholder('label').dtype,
            window.tf_dtypes['label'])
        self.assertAllEqual(
            window.coordinates_placeholder('label').shape.as_list(),
            [1, 7])
        self.assertAllEqual(
            window.coordinates_placeholder('label').dtype, tf.int32)

        window = ImageWindow.from_data_reader_properties(
            **get_all_dynamic_image_window())
        window.placeholders_dict(10)

        self.assertAllEqual(
            window.image_data_placeholder('image').shape.as_list(),
            [1, None, None, None, 1, 2])
        self.assertAllEqual(
            window.image_data_placeholder('image').dtype,
            window.tf_dtypes['image'])
        self.assertAllEqual(
            window.coordinates_placeholder('image').shape.as_list(),
            [1, 7])
        self.assertAllEqual(
            window.coordinates_placeholder('image').dtype, tf.int32)

        self.assertAllEqual(
            window.image_data_placeholder('label').shape.as_list(),
            [1, None, None, None, 1, 1])
        self.assertAllEqual(
            window.image_data_placeholder('label').dtype,
            window.tf_dtypes['label'])
        self.assertAllEqual(
            window.coordinates_placeholder('label').shape.as_list(),
            [1, 7])
        self.assertAllEqual(
            window.coordinates_placeholder('label').dtype, tf.int32)

    def test_set_spatial_size(self):
        new_shape = (42, 43, 44)
        window = ImageWindow.from_data_reader_properties(
            **get_static_window_param())
        window.placeholders_dict(10)
        window.set_spatial_shape(new_shape)
        self.assertAllClose(window.shapes['image'], (10, 42, 43, 44, 1, 2))
        self.assertAllClose(window.shapes['label'], (10, 42, 43, 44, 1, 1))
        window.placeholders_dict(10)

        self.assertAllEqual(
            window.image_data_placeholder('image').shape.as_list(),
            [10, 42, 43, 44, 1, 2])
        self.assertAllEqual(
            window.image_data_placeholder('image').dtype,
            window.tf_dtypes['image'])
        self.assertAllEqual(
            window.coordinates_placeholder('image').shape.as_list(),
            [10, 7])
        self.assertAllEqual(
            window.coordinates_placeholder('image').dtype, tf.int32)

        self.assertAllEqual(
            window.image_data_placeholder('label').shape.as_list(),
            [10, 42, 43, 44, 1, 1])
        self.assertAllEqual(
            window.image_data_placeholder('label').dtype,
            window.tf_dtypes['label'])
        self.assertAllEqual(
            window.coordinates_placeholder('label').shape.as_list(),
            [10, 7])
        self.assertAllEqual(
            window.coordinates_placeholder('label').dtype, tf.int32)


if __name__ == "__main__":
    tf.test.main()
