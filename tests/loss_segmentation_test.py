# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from niftynet.layer.loss_segmentation import LossFunction, labels_to_one_hot
from tests.niftynet_testcase import NiftyNetTestCase


class DiceWithMissingClass(NiftyNetTestCase):
    # all dice methods should return 0.0 for this case:
    def test_missing_class(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 0, 1], [1, 0, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([2, 0], dtype=tf.int32)

            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            for loss_func in ('Dice', 'Dice_NS', 'Dice_Dense', 'Dice_Dense_NS'):

                test_loss_func = LossFunction(3, loss_type=loss_func, softmax=False)
                if 'Dense' in loss_func:
                    loss_value = test_loss_func(predicted, tf.one_hot(labels, 3))
                else:
                    loss_value = test_loss_func(predicted, labels)

                # softmax of zero, Dice loss of -1, so sum \approx -1
                self.assertAllClose(loss_value.eval(), 0.0, atol=1e-4)

    def test_missing_class_dice_plus_xent(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 0, 999], [999, 0, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([2, 0], dtype=tf.int32)

            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            test_loss_func = LossFunction(3, loss_type='DicePlusXEnt', softmax=False)
            loss_value = test_loss_func(predicted, labels)

            # softmax of zero, Dice loss of -1, so sum \approx -1
            self.assertAllClose(loss_value.eval(), -1.0, atol=1e-4)


class DicePlusXEntTest(NiftyNetTestCase):
    def test_dice_plus(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 9999], [9999, 0], [9999, 0], [9999, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int16, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]
            test_loss_func = LossFunction(2, loss_type='DicePlusXEnt', softmax=False)
            loss_value = test_loss_func(predicted, labels)

            # softmax of zero, Dice loss of -1, so sum \approx -1
            self.assertAllClose(loss_value.eval(), -1.0, atol=1e-3)

    def test_dice_plus_multilabel(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 0, 9999], [9999, 0, 0], [0, 9999, 0], [9999, 0, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([2, 0, 1, 0], dtype=tf.int16, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]
            test_loss_func = LossFunction(3, loss_type='DicePlusXEnt', softmax=False)
            loss_value = test_loss_func(predicted, labels)

            # cross-ent of zero, Dice loss of -1, so sum \approx -1
            self.assertAllClose(loss_value.eval(), -1.0, atol=1e-3)

    def test_dice_plus_non_zeros(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 9999, 9999], [9999, 0, 0], [0, 9999, 9999], [9999, 0, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([2, 0, 1, 0], dtype=tf.int16, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]
            test_loss_func = LossFunction(3, loss_type='DicePlusXEnt', softmax=False)
            loss_value = test_loss_func(predicted, labels)
            # cross-ent of mean(ln(2), 0, 0, ln(2)) = .5*ln(2)
            # Dice loss of -mean(1, .5, .5)=-2/3
            self.assertAllClose(loss_value.eval(), .5 * np.log(2) - 2. / 3., atol=1e-3)

    def test_dice_plus_wrong_softmax(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 9999, 9999], [9999, 0, 0], [0, 9999, 9999], [9999, 0, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([2, 0, 1, 0], dtype=tf.int16, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]
            test_loss_func = LossFunction(3, loss_type='DicePlusXEnt', softmax=True)
            loss_value = test_loss_func(predicted, labels)
            # cross-ent of mean(ln(2), 0, 0, ln(2)) = .5*ln(2)
            # Dice loss of -mean(1, .5, .5)=-2/3
            self.assertAllClose(loss_value.eval(), .5 * np.log(2) - 2. / 3., atol=1e-3)

    def test_dice_plus_weighted(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 9999, 9999], [9999, 0, 0], [0, 9999, 9999], [9999, 0, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([2, 0, 1, 0], dtype=tf.int16, name='labels')
            weights = tf.expand_dims(tf.constant([0, 1, 0, 0], dtype=tf.float32), axis=0)
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]
            test_loss_func = LossFunction(3, loss_type='DicePlusXEnt', softmax=True)
            loss_value = test_loss_func(predicted, labels, weight_map=weights)
            self.assertAllClose(loss_value.eval(), -1.)


class OneHotTester(NiftyNetTestCase):
    def test_vs_tf_onehot(self):
        with self.cached_session():
            labels = tf.constant([1, 2, 3, 0], dtype=tf.int64, name='labels')
            tf_one_hot = tf.one_hot(labels, depth=4)
            niftynet_one_hot = tf.sparse_tensor_to_dense(labels_to_one_hot(labels, 4))
            self.assertAllEqual(tf_one_hot.eval(), niftynet_one_hot.eval())

    def test_one_hot(self):
        ref = np.asarray(
            [[[0., 1., 0., 0., 0.], [0., 0., 1., 0., 0.]],
             [[0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]]],
            dtype=np.float32)

        with self.cached_session():
            labels = tf.constant([[1, 2], [3, 4]])
            # import pdb; pdb.set_trace()
            one_hot = tf.sparse_tensor_to_dense(
                labels_to_one_hot(labels, 5)).eval()
            self.assertAllEqual(one_hot, ref)


class SensitivitySpecificityTests(NiftyNetTestCase):
    # test done by regression for refactoring purposes
    def test_sens_spec_loss_by_regression(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            test_loss_func = LossFunction(2, loss_type='SensSpec')
            test_loss = test_loss_func(predicted, labels)
            self.assertAlmostEqual(test_loss.eval(), 2.06106e-9)

    def test_multi_label_sens_spec(self):
        with self.cached_session():
            # answer calculated by hand -
            predicted = tf.constant(
                [[0, 1, 0], [0, 0, 1]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 2], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            test_loss_func = LossFunction(3, loss_type='SensSpec',
                                          loss_func_params={'r': 0.05})
            test_loss = test_loss_func(predicted, labels)
            self.assertAlmostEqual(test_loss.eval(), 0.14598623)


class GeneralisedDiceTest(NiftyNetTestCase):
    # test done by regression for refactoring purposes
    def test_generalised_dice_score_regression(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            test_loss_func = LossFunction(2, loss_type='GDSC')
            one_minus_generalised_dice_score = test_loss_func(
                predicted, labels)
            self.assertAllClose(
                one_minus_generalised_dice_score.eval(), 0.0, atol=1e-4)

    def test_gdsc_incorrect_type_weight_error(self):
        with self.cached_session():
            with self.assertRaises(ValueError) as cm:
                predicted = tf.constant(
                    [[0, 10], [10, 0], [10, 0], [10, 0]],
                    dtype=tf.float32, name='predicted')
                labels = tf.constant(
                    [1, 0, 0, 0],
                    dtype=tf.int64, name='labels')
                predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

                test_loss_func = LossFunction(
                    2, loss_type='GDSC',
                    loss_func_params={'type_weight': 'unknown'})
                one_minus_generalised_dice_score = test_loss_func(predicted,
                                                                  labels)

    def test_generalised_dice_score_uniform_regression(self):
        with self.cached_session():
            predicted = tf.constant([[0, 10], [10, 0], [10, 0], [10, 0]],
                                    dtype=tf.float32, name='predicted')

            labels = tf.constant([[1, 0, 0, 0]], dtype=tf.int64, name='labels')
            weights = tf.cast(labels, tf.float32)
            predicted, labels, weights = [tf.expand_dims(x, axis=0) for x in
                                          (predicted, labels, weights)]

            test_loss_func = LossFunction(
                2, loss_type='GDSC',
                loss_func_params={'type_weight': 'Uniform'})
            one_minus_generalised_dice_score = test_loss_func(
                predicted, labels, weights)
            self.assertAllClose(one_minus_generalised_dice_score.eval(),
                                0.3333, atol=1e-4)


class DiceTest(NiftyNetTestCase):
    def test_dice_score(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            test_loss_func = LossFunction(2, loss_type='Dice')
            one_minus_dice_score = test_loss_func(predicted, labels)
            self.assertAllClose(one_minus_dice_score.eval(), 0.0, atol=1e-5)

    def test_dice_score_weights(self):
        with self.cached_session():
            weights = tf.constant([[1, 1, 0, 0]], dtype=tf.float32,
                                  name='weights')
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([[1, 0, 0, 0]], dtype=tf.int64, name='labels')
            predicted, labels, weights = [tf.expand_dims(x, axis=0) for x in (predicted, labels, weights)]

            test_loss_func = LossFunction(2, loss_type='Dice')
            one_minus_dice_score = test_loss_func(predicted, labels,
                                                  weight_map=weights)
            self.assertAllClose(one_minus_dice_score.eval(), 0.0, atol=1e-4)

    def test_wrong_prediction(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 100]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([0], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            test_loss_func = LossFunction(2, loss_type='Dice')
            one_minus_dice_score = test_loss_func(predicted, labels)
            self.assertAllClose(one_minus_dice_score.eval(), 1.0, atol=1e-4)

    def test_dice_batch_size_greater_than_one(self):
        # test for Github issue #22: need to take mean per-image before
        # averaging Dice of ~1 and ~0.16, should get dice ~ 1 - 0.5816
        with self.cached_session():
            # predictions ~ [1, 0, 0]; [0, 0, 1]; [0, .5, .5]; [.333, .333, .333]
            predictions_numpy = np.array([[[10., 0, 0], [0, 0, 10]],
                                          [[-10, 0, 0], [0, 0, 0]]]).reshape([2, 2, 1, 1, 3])
            labels_numpy = np.array([[[0, 2]], [[0, 1]]]).reshape([2, 2, 1, 1, 1])

            predicted = tf.constant(predictions_numpy, dtype=tf.float32, name='predicted')
            labels = tf.constant(labels_numpy, dtype=tf.int64, name='labels')

            test_loss_func = LossFunction(3, loss_type='Dice')
            one_minus_dice_score = test_loss_func(predicted, labels)

            self.assertAllClose(one_minus_dice_score.eval(), 1 - 0.5816, atol=1e-4)


class CrossEntropyTests(NiftyNetTestCase):
    def test_cross_entropy_value(self):
        # test value is -0.5 * [1 * log(e / (1+e)) + 1 * log(e^2 / (e^2 + 1))]
        with self.cached_session():
            predicted = tf.constant(
                [[0, 1], [2, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            test_loss_func = LossFunction(2, loss_type='CrossEntropy')
            computed_cross_entropy = test_loss_func(predicted, labels)
            self.assertAlmostEqual(
                computed_cross_entropy.eval(),
                -.5 * (np.log(np.e / (1 + np.e)) + np.log(
                    np.e ** 2 / (1 + np.e ** 2))))

            test_dense_loss = LossFunction(2, loss_type='CrossEntropy_Dense')
            labels = tf.sparse_tensor_to_dense(labels_to_one_hot(labels, 2))
            computed_cross_entropy = test_loss_func(predicted, tf.to_int32(labels))
            self.assertAlmostEqual(
                computed_cross_entropy.eval(),
                -.5 * (np.log(np.e / (1 + np.e)) + np.log(
                    np.e ** 2 / (1 + np.e ** 2))))

    def test_cross_entropy_value_weight(self):
        with self.cached_session():
            weights = tf.constant([[1], [2]], dtype=tf.float32, name='weights')
            predicted = tf.constant(
                [[0, 1], [2, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([[1], [0]], dtype=tf.int64, name='labels')
            predicted, labels, weights = \
                [tf.expand_dims(x, axis=0) for x in (predicted, labels, weights)]

            test_loss_func = LossFunction(2, loss_type='CrossEntropy')
            computed_cross_entropy = test_loss_func(predicted, labels, weights)
            self.assertAlmostEqual(
                computed_cross_entropy.eval(),
                -.5 * (
                        2.0 / 3.0 * np.log(np.e / (1 + np.e)) + 4.0 / 3.0 * np.log(
                    np.e ** 2 / (1 + np.e ** 2))))


class DiceTestNoSquare(NiftyNetTestCase):
    def test_dice_score_nosquare(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            test_loss_func = LossFunction(2, loss_type='Dice_NS')
            one_minus_dice_score = test_loss_func(predicted, labels)
            self.assertAllClose(one_minus_dice_score.eval(), 0.0, atol=1e-4)

    def test_dice_score_nosquare_weights(self):
        with self.cached_session():
            weights = tf.constant([[1, 1, 0, 0]], dtype=tf.float32,
                                  name='weights')
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([[1, 0, 0, 0]], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            test_loss_func = LossFunction(2,
                                          loss_type='Dice_NS')
            one_minus_dice_score = test_loss_func(predicted, labels,
                                                  weight_map=weights)
            self.assertAllClose(one_minus_dice_score.eval(), 0.0, atol=1e-4)

    def test_wrong_prediction(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 100]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([0], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            test_loss_func = LossFunction(2, loss_type='Dice_NS')
            one_minus_dice_score = test_loss_func(predicted, labels)
            self.assertAllClose(one_minus_dice_score.eval(), 1.0, atol=1e-4)


class TverskyTest(NiftyNetTestCase):
    def test_tversky_index(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0)
                                 for x in (predicted, labels)]

            test_loss_func = LossFunction(2, loss_type='Tversky')
            one_minus_tversky_index = test_loss_func(predicted, labels)
            self.assertAllClose(one_minus_tversky_index.eval(), 0.0, atol=1e-4)

    def test_tversky_index_weights(self):
        with self.cached_session():
            weights = tf.constant([[1, 1, 0, 0]], dtype=tf.float32,
                                  name='weights')
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([[1, 0, 1, 0]], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0)
                                 for x in (predicted, labels)]

            test_loss_func = LossFunction(2,
                                          loss_type='Tversky')
            one_minus_tversky_index = test_loss_func(predicted, labels,
                                                     weight_map=weights)
            self.assertAllClose(one_minus_tversky_index.eval(), 0.0, atol=1e-4)

    def test_wrong_prediction(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 100]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([0], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0)
                                 for x in (predicted, labels)]

            test_loss_func = LossFunction(2, loss_type='Tversky')
            one_minus_tversky_index = test_loss_func(predicted, labels)
            self.assertAlmostEqual(one_minus_tversky_index.eval(), 1.0)


class DiceDenseTest(NiftyNetTestCase):
    def test_dice_dense_score(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 10], [10, 0], [10, 0], [10, 0]],
                dtype=tf.float32, name='predicted')
            one_hot = tf.constant([[1, 0], [0, 1], [0, 1], [0, 1]],
                                  dtype=tf.int64, name='one_hot')
            predicted, one_hot = [tf.expand_dims(x, axis=0) for x in (predicted, one_hot)]

            test_loss_func = LossFunction(2, loss_type='Dice_Dense')
            one_minus_dice_score = test_loss_func(predicted, one_hot)
            self.assertAllClose(one_minus_dice_score.eval(), 1.0, atol=1e-4)

    def test_wrong_prediction(self):
        with self.cached_session():
            predicted = tf.constant(
                [[0, 100]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([0], dtype=tf.int64, name='labels')
            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]
            one_hot = tf.one_hot(labels, axis=-1, depth=2)

            test_loss_func = LossFunction(2, loss_type='Dice_Dense')
            one_minus_dice_score = test_loss_func(predicted, one_hot)
            self.assertAllClose(one_minus_dice_score.eval(), 1.0, atol=1e-4)

    def test_dense_dice_vs_sparse(self):
        # regression test vs dense version
        with self.cached_session():
            predicted = tf.constant(
                [[2, 3], [9, 8], [0, 0], [1, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')

            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            sparse_loss_func = LossFunction(2, loss_type='Dice')
            sparse_dice = sparse_loss_func(predicted, labels)

            one_hot = tf.one_hot(labels, axis=-1, depth=2)
            dense_loss_func = LossFunction(2, loss_type='Dice_Dense')
            dense_dice = dense_loss_func(predicted, one_hot)

            self.assertAllEqual(sparse_dice.eval(), dense_dice.eval())


class DiceDenseNoSquareTest(NiftyNetTestCase):

    def test_dense_dice_nosquare_vs_sparse(self):
        # regression test vs dense version
        with self.cached_session():
            predicted = tf.constant(
                [[2, 3], [9, 8], [0, 0], [1, 0]],
                dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')

            predicted, labels = [tf.expand_dims(x, axis=0) for x in (predicted, labels)]

            sparse_loss_func = LossFunction(2, loss_type='Dice_NS')
            sparse_dice = sparse_loss_func(predicted, labels)

            one_hot = tf.one_hot(labels, axis=-1, depth=2)
            dense_loss_func = LossFunction(2, loss_type='Dice_Dense_NS')
            dense_dice = dense_loss_func(predicted, one_hot)

            self.assertAllEqual(sparse_dice.eval(), dense_dice.eval())


class VolumeEnforcementTest(NiftyNetTestCase):
    def test_volume_enforcement_equal(self):
        with self.cached_session():
            predicted = tf.constant([[-1000, 1000], [1000, -1000], [1000,
                                                                     -1000], [1000, -1000]],
                                    dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')

            predicted, labels = [tf.expand_dims(x, axis=0) for x in
                                 (predicted, labels)]
            venf_loss_func = LossFunction(2, loss_type='VolEnforcement')
            venf_loss = venf_loss_func(predicted, labels)
            self.assertAllClose(venf_loss.eval(), 0.0, atol=1e-4)

    def test_volume_enforcement_nonexist(self):
        with self.cached_session():
            predicted = tf.constant([[1000, -1000], [1000, -1000], [1000, -1000], [1000, -1000]],
                                    dtype=tf.float32, name='predicted')
            labels = tf.constant([1, 0, 0, 0], dtype=tf.int64, name='labels')

            predicted, labels = [tf.expand_dims(x, axis=0) for x in
                                 (predicted, labels)]
            venf_loss_func = LossFunction(2, loss_type='VolEnforcement')
            venf_loss = venf_loss_func(predicted, labels)
            self.assertAllClose(venf_loss.eval(), 500.75, atol=0.1)







class LossFunctionErrorsTest(NiftyNetTestCase):
    """
    These tests check that a ValueError is called
    for non-existent loss functions.
    They also check that suggestions are returned
    if the name is close to a real one.
    """

    def test_value_error_for_bad_loss_function(self):
        with self.cached_session():
            with self.assertRaises(ValueError):
                LossFunction(1, loss_type='wrong answer')

    # Note: sensitive to precise wording of ValueError message.
    def test_suggestion_for_dice_typo(self):
        with self.cached_session():
            with self.assertRaisesRegexp(ValueError, 'Dice'):
                LossFunction(1, loss_type='dice')

    def test_suggestion_for_gdsc_typo(self):
        with self.cached_session():
            with self.assertRaisesRegexp(ValueError, 'GDSC'):
                LossFunction(1, loss_type='GSDC')


if __name__ == '__main__':
    tf.test.main()
