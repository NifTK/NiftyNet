# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

# Before doing anything else, check TF is installed
# and fail gracefully if not.
try:
    import tensorflow
except ImportError:
    raise ImportError('NiftyNet is based on TensorFlow, which'
                      ' does not seem to be installed on your'
                      ' system.\nPlease install TensorFlow'
                      ' (https://www.tensorflow.org/) to be'
                      ' able to use NiftyNet.')

import os

import niftynet.utilities.misc_common as util
import niftynet.utilities.parse_user_params as parse_user_params
from niftynet.engine.volume_loader import VolumeLoaderLayer
from niftynet.layer.binary_masking import BinaryMaskingLayer
from niftynet.layer.histogram_normalisation import \
    HistogramNormalisationLayer
from niftynet.layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer
from niftynet.utilities.csv_table import CSVTable


# if sys.version_info[0] >= 3:
#    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')
# else:
#    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)



class NetFactory(object):
    @staticmethod
    def create(name):
        if name == "highres3dnet":
            from niftynet.network.highres3dnet import HighRes3DNet
            return HighRes3DNet
        if name == "highres3dnet_small":
            from niftynet.network.highres3dnet_small import HighRes3DNetSmall
            return HighRes3DNetSmall
        if name == "highres3dnet_large":
            from niftynet.network.highres3dnet_large import HighRes3DNetLarge
            return HighRes3DNetLarge
        elif name == "toynet":
            from niftynet.network.toynet import ToyNet
            return ToyNet
        elif name == "unet":
            from niftynet.network.unet import UNet3D
            return UNet3D
        elif name == "vnet":
            from niftynet.network.vnet import VNet
            return VNet
        elif name == "dense_vnet":
            from niftynet.network.dense_vnet import DenseVNet
            return DenseVNet
        elif name == "deepmedic":
            from niftynet.network.deepmedic import DeepMedic
            return DeepMedic
        elif name == "scalenet":
            from niftynet.network.scalenet import ScaleNet
            return ScaleNet
        else:
            print("network: \"{}\" not implemented".format(name))
            raise NotImplementedError


def main():
    param, csv_dict = parse_user_params.run()
    if util.has_bad_inputs(param):
        return -1
    else:
        util.print_save_input_parameters(param, ini_file=os.path.join(param.model_dir,
                                                                      'settings_' + param.action + '.ini'))

    if not (param.cuda_devices == '""'):
        os.environ["CUDA_VISIBLE_DEVICES"] = param.cuda_devices
        print("set CUDA_VISIBLE_DEVICES env to {}".format(param.cuda_devices))
    net_class = NetFactory.create(param.net_name)

    # expanding a few of the user input parameters
    if param.spatial_rank == 3:
        spatial_padding = \
            ((param.volume_padding_size, param.volume_padding_size),
             (param.volume_padding_size, param.volume_padding_size),
             (param.volume_padding_size, param.volume_padding_size))
    else:
        spatial_padding = \
            ((param.volume_padding_size, param.volume_padding_size),
             (param.volume_padding_size, param.volume_padding_size))
    interp_order = (param.image_interp_order,
                    param.label_interp_order,
                    param.w_map_interp_order)

    # read each line of csv files into an instance of Subject
    csv_loader = CSVTable(csv_dict=csv_dict, allow_missing=True)

    # define layers of volume-level normalisation
    normalisation_layers = []
    if param.normalisation:
        hist_norm = HistogramNormalisationLayer(
            models_filename=param.histogram_ref_file,
            binary_masking_func=BinaryMaskingLayer(
                type=param.mask_type,
                multimod_fusion=param.multimod_mask_type),
            norm_type=param.norm_type,
            cutoff=(param.cutoff_min, param.cutoff_max))
        normalisation_layers.append(hist_norm)
    if param.whitening:
        mean_std_norm = MeanVarNormalisationLayer(
            binary_masking_func=BinaryMaskingLayer(
                type=param.mask_type,
                multimod_fusion=param.multimod_mask_type))
        normalisation_layers.append(mean_std_norm)

    # define how to load image volumes
    volume_loader = VolumeLoaderLayer(
        csv_loader,
        standardisor=normalisation_layers,
        is_training=(param.action == "train"),
        do_reorientation=param.reorientation,
        do_resampling=param.resampling,
        spatial_padding=spatial_padding,
        interp_order=interp_order)
    print('found {} subjects'.format(len(volume_loader.subject_list)))

    if param.action == "train":
        import niftynet.engine.training

        device_str = "gpu" if param.num_gpus > 0 else "cpu"
        niftynet.engine.training.run(net_class, param, volume_loader, device_str)
    else:
        import niftynet.engine.inference

        device_str = "gpu" if param.num_gpus > 0 else "cpu"
        niftynet.engine.inference.run(net_class, param, volume_loader, device_str)
    return 0

