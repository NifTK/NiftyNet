# -*- coding: utf-8 -*-
"""
Loading modules from a string representing the class name
or a short name that matches the dictionary item defined
in this module
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import importlib
import os

import tensorflow as tf

from niftynet.utilities.util_common import \
    damerau_levenshtein_distance as edit_distance

# pylint: disable=too-few-public-methods
SUPPORTED_APP = {
    'net_regress':
        'niftynet.application.regression_application.RegressionApplication',
    'net_segment':
        'niftynet.application.segmentation_application.SegmentationApplication',
    'net_autoencoder':
        'niftynet.application.autoencoder_application.AutoencoderApplication',
    'net_gan':
        'niftynet.application.gan_application.GANApplication',
    'net_classify':
        'niftynet.application.classification_application.'
        'ClassificationApplication',
}

SUPPORTED_NETWORK = {
    # GAN
    'simulator_gan':
        'niftynet.network.simulator_gan.SimulatorGAN',
    'simple_gan':
        'niftynet.network.simple_gan.SimpleGAN',

    # Segmentation
    "highres3dnet":
        'niftynet.network.highres3dnet.HighRes3DNet',
    "highres3dnet_small":
        'niftynet.network.highres3dnet_small.HighRes3DNetSmall',
    "highres3dnet_large":
        'niftynet.network.highres3dnet_large.HighRes3DNetLarge',
    "toynet":
        'niftynet.network.toynet.ToyNet',
    "unet":
        'niftynet.network.unet.UNet3D',
    "nonewnet":
        'niftynet.network.no_new_net.UNet3D',
    "vnet":
        'niftynet.network.vnet.VNet',
    "dense_vnet":
        'niftynet.network.dense_vnet.DenseVNet',
    "deepmedic":
        'niftynet.network.deepmedic.DeepMedic',
    "scalenet":
        'niftynet.network.scalenet.ScaleNet',
    "holisticnet":
        'niftynet.network.holistic_net.HolisticNet',
    "unet_2d":
        'niftynet.network.unet_2d.UNet2D',

    # classification
    "resnet": 'niftynet.network.resnet.ResNet',
    "se_resnet": 'niftynet.network.se_resnet.SE_ResNet',

    # autoencoder
    "vae": 'niftynet.network.vae.VAE'
}

SUPPORTED_LOSS_GAN = {
    'CrossEntropy': 'niftynet.layer.loss_gan.cross_entropy',
}

SUPPORTED_LOSS_SEGMENTATION = {
    "CrossEntropy":
        'niftynet.layer.loss_segmentation.cross_entropy',
    "CrossEntropy_Dense":
        'niftynet.layer.loss_segmentation.cross_entropy_dense',
    "Dice":
        'niftynet.layer.loss_segmentation.dice',
    "Dice_NS":
        'niftynet.layer.loss_segmentation.dice_nosquare',
    "Dice_Dense":
        'niftynet.layer.loss_segmentation.dice_dense',
    "Dice_Dense_NS":
        'niftynet.layer.loss_segmentation.dice_dense_nosquare',
    "Tversky":
        'niftynet.layer.loss_segmentation.tversky',
    "GDSC":
        'niftynet.layer.loss_segmentation.generalised_dice_loss',
    "DicePlusXEnt":
        'niftynet.layer.loss_segmentation.dice_plus_xent_loss',
    "WGDL":
        'niftynet.layer.loss_segmentation.generalised_wasserstein_dice_loss',
    "SensSpec":
        'niftynet.layer.loss_segmentation.sensitivity_specificity_loss',
    "VolEnforcement":
        'niftynet.layer.loss_segmentation.volume_enforcement',
    # "L1Loss":
    #     'niftynet.layer.loss_segmentation.l1_loss',
    # "L2Loss":
    #     'niftynet.layer.loss_segmentation.l2_loss',
    # "Huber":
    #     'niftynet.layer.loss_segmentation.huber_loss'
}

SUPPORTED_LOSS_REGRESSION = {
    "L1Loss":
        'niftynet.layer.loss_regression.l1_loss',
    "L2Loss":
        'niftynet.layer.loss_regression.l2_loss',
    "RMSE":
        'niftynet.layer.loss_regression.rmse_loss',
    "MAE":
        'niftynet.layer.loss_regression.mae_loss',
    "Huber":
        'niftynet.layer.loss_regression.huber_loss',
    "SmoothL1":
        'niftynet.layer.loss_regression.smooth_l1_loss',
    "Cosine":
        'niftynet.layer.loss_regression.cosine_loss'
}

SUPPORTED_LOSS_CLASSIFICATION = {
    "CrossEntropy":
        'niftynet.layer.loss_classification.cross_entropy',
}

SUPPORTED_LOSS_CLASSIFICATION_MULTI = {
    "ConfusionMatrix":
        'niftynet.layer.loss_classification_multi.loss_confusion_matrix',
    "Variability":
        'niftynet.layer.loss_classification_multi.loss_variability',
    "Consistency":
        'niftynet.layer.loss_classification_multi.rmse_consistency'
}


SUPPORTED_LOSS_AUTOENCODER = {
    "VariationalLowerBound":
        'niftynet.layer.loss_autoencoder.variational_lower_bound',
}

SUPPORTED_OPTIMIZERS = {
    'adam': 'niftynet.engine.application_optimiser.Adam',
    'gradientdescent': 'niftynet.engine.application_optimiser.GradientDescent',
    'momentum': 'niftynet.engine.application_optimiser.Momentum',
    'nesterov': 'niftynet.engine.application_optimiser.NesterovMomentum',

    'adagrad': 'niftynet.engine.application_optimiser.Adagrad',
    'rmsprop': 'niftynet.engine.application_optimiser.RMSProp',
}

SUPPORTED_INITIALIZATIONS = {
    'constant': 'niftynet.engine.application_initializer.Constant',
    'zeros': 'niftynet.engine.application_initializer.Zeros',
    'ones': 'niftynet.engine.application_initializer.Ones',
    'uniform_scaling':
        'niftynet.engine.application_initializer.UniformUnitScaling',
    'orthogonal': 'niftynet.engine.application_initializer.Orthogonal',
    'variance_scaling':
        'niftynet.engine.application_initializer.VarianceScaling',
    'glorot_normal':
        'niftynet.engine.application_initializer.GlorotNormal',
    'glorot_uniform':
        'niftynet.engine.application_initializer.GlorotUniform',
    'he_normal': 'niftynet.engine.application_initializer.HeNormal',
    'he_uniform': 'niftynet.engine.application_initializer.HeUniform'
}

SUPPORTED_EVALUATIONS = {
    'dice': 'niftynet.evaluation.segmentation_evaluations.dice',
    'jaccard': 'niftynet.evaluation.segmentation_evaluations.jaccard',
    'Dice': 'niftynet.evaluation.segmentation_evaluations.dice',
    'Jaccard': 'niftynet.evaluation.segmentation_evaluations.jaccard',
    'n_pos_ref': 'niftynet.evaluation.segmentation_evaluations.n_pos_ref',
    'n_neg_ref': 'niftynet.evaluation.segmentation_evaluations.n_neg_ref',
    'n_pos_seg': 'niftynet.evaluation.segmentation_evaluations.n_pos_seg',
    'n_neg_seg': 'niftynet.evaluation.segmentation_evaluations.n_neg_seg',
    'fp': 'niftynet.evaluation.segmentation_evaluations.fp',
    'fn': 'niftynet.evaluation.segmentation_evaluations.fn',
    'tp': 'niftynet.evaluation.segmentation_evaluations.tp',
    'tn': 'niftynet.evaluation.segmentation_evaluations.tn',
    'n_intersection': 'niftynet.evaluation.segmentation_evaluations'
                      '.n_intersection',
    'n_union': 'niftynet.evaluation.segmentation_evaluations.n_union',
    'specificity': 'niftynet.evaluation.segmentation_evaluations.specificity',
    'sensitivity': 'niftynet.evaluation.segmentation_evaluations.sensitivity',
    'accuracy': 'niftynet.evaluation.segmentation_evaluations.accuracy',
    'false_positive_rate': 'niftynet.evaluation.segmentation_evaluations'
                           '.false_positive_rate',
    'positive_predictive_values': 'niftynet.evaluation.segmentation_evaluations'
                                  '.positive_predictive_values',
    'negative_predictive_values': 'niftynet.evaluation.segmentation_evaluations'
                                  '.negative_predictive_values',
    'intersection_over_union': 'niftynet.evaluation.segmentation_evaluations'
                               '.intersection_over_union',
    'informedness': 'niftynet.evaluation.segmentation_evaluations.informedness',
    'markedness': 'niftynet.evaluation.segmentation_evaluations.markedness',
    'vol_diff': 'niftynet.evaluation.segmentation_evaluations.vol_diff',
    'average_distance': 'niftynet.evaluation.segmentation_evaluations'
                        '.average_distance',
    'hausdorff_distance': 'niftynet.evaluation.segmentation_evaluations'
                          '.hausdorff_distance',
    'hausdorff95_distance': 'niftynet.evaluation.segmentation_evaluations'
                            '.hausdorff95_distance',
    'com_ref': 'niftynet.contrib.evaluation.segmentation_evaluations.com_ref',
    'mse': 'niftynet.evaluation.regression_evaluations.mse',
    'rmse': 'niftynet.evaluation.regression_evaluations.rmse',
    'mae': 'niftynet.evaluation.regression_evaluations.mae',
    # 'r2': 'niftynet.contrib.evaluation.regression_evaluations.r2',
    'classification_accuracy': 'niftynet.evaluation.classification_evaluations'
                               '.accuracy',
    'roc_auc': 'niftynet.contrib.evaluation.classification_evaluations.roc_auc',
    'roc': 'niftynet.contrib.evaluation.classification_evaluations.roc',
}

SUPPORTED_EVENT_HANDLERS = {
    'model_restorer':
        'niftynet.engine.handler_model.ModelRestorer',
    'model_saver':
        'niftynet.engine.handler_model.ModelSaver',
    'sampler_threading':
        'niftynet.engine.handler_sampler.SamplerThreading',
    'apply_gradients':
        'niftynet.engine.handler_gradient.ApplyGradients',
    'output_interpreter':
        'niftynet.engine.handler_network_output.OutputInterpreter',
    'console_logger':
        'niftynet.engine.handler_console.ConsoleLogger',
    'tensorboard_logger':
        'niftynet.engine.handler_tensorboard.TensorBoardLogger',
    'performance_logger':
        'niftynet.engine.handler_performance.PerformanceLogger',
    'early_stopper':
        'niftynet.engine.handler_early_stopping.EarlyStopper',
}

SUPPORTED_ITERATION_GENERATORS = {
    'iteration_generator':
        'niftynet.engine.application_iteration.IterationMessageGenerator'
}


def select_module(module_name, type_str, lookup_table=None):
    """
    This function first tries to find the absolute module name
    by matching the static dictionary items, if not found, it
    tries to import the module by splitting the input ``module_name``
    as module name and class name to be imported.

    :param module_name: string that matches the keys defined in lookup_table
        or an absolute class name: module.name.ClassName
    :param type_str: type of the module (used for better error display)
    :param lookup_table: defines a set of shorthands for absolute class name
    """
    lookup_table = lookup_table or {}
    module_name = '{}'.format(module_name)
    is_external = True
    if module_name in lookup_table:
        module_name = lookup_table[module_name]
        is_external = False
    module_str, class_name = None, None
    try:
        module_str, class_name = module_name.rsplit('.', 1)
        the_module = importlib.import_module(module_str)
        the_class = getattr(the_module, class_name)
        if is_external:
            # print location of external module
            tf.logging.info('Import [%s] from %s.',
                            class_name, os.path.abspath(the_module.__file__))
        return the_class
    except (AttributeError, ValueError, ImportError) as not_imported:
        tf.logging.fatal(repr(not_imported))
        if '.' not in module_name:
            err = 'Could not import {}: ' \
                  'Incorrect module name "{}"; ' \
                  'expected "module.object".'.format(type_str, module_name)
        else:
            err = '{}: Could not import object' \
                  '"{}" from "{}"'.format(type_str, class_name, module_str)
        tf.logging.fatal(err)

        if not lookup_table:
            # no further guess
            raise ValueError(err)

        dists = dict(
            (k, edit_distance(k, module_name)) for k in list(lookup_table))
        closest = min(dists, key=dists.get)
        if dists[closest] <= 3:
            err = 'Could not import {2}: By "{0}", ' \
                  'did you mean "{1}"?\n "{0}" is ' \
                  'not a valid option. '.format(module_name, closest, type_str)
            tf.logging.fatal(err)
        raise ValueError(err)


class ModuleFactory(object):
    """
    General interface for importing a class by its name.
    """
    SUPPORTED = None
    type_str = 'object'

    @classmethod
    def create(cls, name):
        """
        import a class by name
        """
        return select_module(name, cls.type_str, cls.SUPPORTED)


class ApplicationNetFactory(ModuleFactory):
    """
    Import a network from ``niftynet.network`` or from user specified string
    """
    SUPPORTED = SUPPORTED_NETWORK
    type_str = 'network'


class ApplicationFactory(ModuleFactory):
    """
    Import an application from ``niftynet.application`` or
    from user specified string
    """
    SUPPORTED = SUPPORTED_APP
    type_str = 'application'


class LossGANFactory(ModuleFactory):
    """
    Import a GAN loss function from ``niftynet.layer`` or
    from user specified string
    """
    SUPPORTED = SUPPORTED_LOSS_GAN
    type_str = 'GAN loss'


class LossSegmentationFactory(ModuleFactory):
    """
    Import a segmentation loss function from ``niftynet.layer`` or
    from user specified string
    """
    SUPPORTED = SUPPORTED_LOSS_SEGMENTATION
    type_str = 'segmentation loss'


class LossRegressionFactory(ModuleFactory):
    """
    Import a regression loss function from ``niftynet.layer`` or
    from user specified string
    """
    SUPPORTED = SUPPORTED_LOSS_REGRESSION
    type_str = 'regression loss'


class LossClassificationFactory(ModuleFactory):
    """
    Import a classification loss function from niftynet.layer or
    from user specified string
    """
    SUPPORTED = SUPPORTED_LOSS_CLASSIFICATION
    type_str = 'classification loss'


class LossClassificationMultiFactory(ModuleFactory):
    """
    Import a classification loss function from niftynet.layer or
    from user specified string
    """
    SUPPORTED = SUPPORTED_LOSS_CLASSIFICATION_MULTI
    type_str = 'classification multi loss'


class LossAutoencoderFactory(ModuleFactory):
    """
    Import an autoencoder loss function from ``niftynet.layer`` or
    from user specified string
    """
    SUPPORTED = SUPPORTED_LOSS_AUTOENCODER
    type_str = 'autoencoder loss'


class OptimiserFactory(ModuleFactory):
    """
    Import an optimiser from ``niftynet.engine.application_optimiser`` or
    from user specified string
    """
    SUPPORTED = SUPPORTED_OPTIMIZERS
    type_str = 'optimizer'


class InitializerFactory(ModuleFactory):
    """
    Import an initializer from ``niftynet.engine.application_initializer`` or
    from user specified string
    """
    SUPPORTED = SUPPORTED_INITIALIZATIONS
    type_str = 'initializer'

    @staticmethod
    def get_initializer(name, args=None):
        """
        wrapper for getting the initializer.

        :param name:
        :param args: optional parameters for the initializer
        :return:
        """
        init_class = InitializerFactory.create(name)
        if args is None:
            args = {}
        return init_class.get_instance(args)


class EvaluationFactory(ModuleFactory):
    """
    Import an optimiser from niftynet.engine.application_optimiser or
    from user specified string
    """
    SUPPORTED = SUPPORTED_EVALUATIONS
    type_str = 'evaluation'


class EventHandlerFactory(ModuleFactory):
    """
    Import an event handler such as niftynet.engine.handler_console
    """
    SUPPORTED = SUPPORTED_EVENT_HANDLERS
    type_str = 'event handler'


class IteratorFactory(ModuleFactory):
    """
    Import an iterative message generator for the main engine loop
    """
    SUPPORTED = SUPPORTED_ITERATION_GENERATORS
    type_str = 'engine iterator'
