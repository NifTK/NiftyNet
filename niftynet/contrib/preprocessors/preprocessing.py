from niftynet.layer.mean_variance_normalisation import MeanVarNormalisationLayer
from niftynet.layer.rand_flip import RandomFlipLayer
from niftynet.layer.rand_rotation import RandomRotationLayer
from niftynet.layer.rand_spatial_scaling import RandomSpatialScalingLayer
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.histogram_normalisation import HistogramNormalisationLayer


class Preprocessing:
    """
    This class returns normalisation and augmentation layers for use with reader objects.
    """
    def __init__(self, net_param, action_param, task_param):
        self.net_param = net_param
        self.action_param = action_param
        self.task_param = task_param

    def prepare_normalisation_layers(self):
        """
        returns list of normalisation layers
        """
        foreground_masking_layer = None
        if self.net_param.normalise_foreground_only:
            foreground_masking_layer = BinaryMaskingLayer(
                type_str=self.net_param.foreground_type,
                multimod_fusion=self.net_param.multimod_foreground_type,
                threshold=0.0)

        mean_var_normaliser = MeanVarNormalisationLayer(
            image_name='image',
            binary_masking_func=foreground_masking_layer)
        histogram_normaliser = None
        if self.net_param.histogram_ref_file:
            histogram_normaliser = HistogramNormalisationLayer(
                image_name='image',
                modalities=vars(self.task_param).get('image'),
                model_filename=self.net_param.histogram_ref_file,
                binary_masking_func=foreground_masking_layer,
                norm_type=self.net_param.norm_type,
                cutoff=self.net_param.cutoff,
                name='hist_norm_layer')
        normalisation_layers = []
        if self.net_param.normalisation:
            normalisation_layers.append(histogram_normaliser)
        if self.net_param.whitening:
            normalisation_layers.append(mean_var_normaliser)
        return normalisation_layers

    def prepare_augmentation_layers(self):
        """
        returns list of augmentation layers
        """
        augmentation_layers = []
        if self.action_param.random_flipping_axes != -1:
            augmentation_layers.append(RandomFlipLayer(
                flip_axes=self.action_param.random_flipping_axes))
        if self.action_param.scaling_percentage:
            augmentation_layers.append(RandomSpatialScalingLayer(
                min_percentage=self.action_param.scaling_percentage[0],
                max_percentage=self.action_param.scaling_percentage[1],
                antialiasing=self.action_param.antialiasing))
        if self.action_param.rotation_angle or \
                self.action_param.rotation_angle_x or \
                self.action_param.rotation_angle_y or \
                self.action_param.rotation_angle_z:
            rotation_layer = RandomRotationLayer()
            if self.action_param.rotation_angle:
                rotation_layer.init_uniform_angle(
                    self.action_param.rotation_angle)
            else:
                rotation_layer.init_non_uniform_angle(
                    self.action_param.rotation_angle_x,
                    self.action_param.rotation_angle_y,
                    self.action_param.rotation_angle_z)
            augmentation_layers.append(rotation_layer)
        return augmentation_layers
