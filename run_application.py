# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys

import utilities.misc_common as util
import utilities.parse_user_params as parse_user_params
from engine.volume_loader import VolumeLoaderLayer
from layer.binary_masking import BinaryMaskingLayer
from layer.histogram_normalisation import \
    HistogramNormalisationLayer as HistNorm
from layer.mean_variance_normalisation import \
    MeanVarNormalisationLayer as MVNorm
from utilities.csv_table import CSVTable


# if sys.version_info[0] >= 3:
#    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')
# else:
#    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)



class NetFactory(object):
    @staticmethod
    def create(name):
        if name == "highres3dnet":
            from network.highres3dnet import HighRes3DNet
            return HighRes3DNet
        elif name == "toynet":
            from network.toynet import ToyNet
            return ToyNet
        elif name == "unet":
            from network.unet import UNet3D
            return UNet3D
        elif name == "vnet":
            from network.vnet import VNet
            return VNet
        elif name == "deepmedic":
            from network.deepmedic import DeepMedic
            return DeepMedic
        elif name == "scalenet":
            from network.scalenet import ScaleNet
            return ScaleNet
        elif name == "autoencoder_basic":
            from network.autoencoder_basic import AutoEncoderBasic
            return AutoEncoderBasic
        elif name == "vae_basic":
            from network.vae_basic import VAE_basic
            return VAE_basic
        else:
            print("network: \"{}\" not implemented".format(name))
            raise NotImplementedError


if __name__ == "__main__":
    param, csv_dict = parse_user_params.run()
    if util.has_bad_inputs(param):
        sys.exit(-1)
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
        hist_norm = HistNorm(
            models_filename=param.histogram_ref_file,
            binary_masking_func=BinaryMaskingLayer(
                type=param.mask_type,
                multimod_fusion=param.multimod_mask_type),
            norm_type=param.norm_type,
            cutoff=(param.cutoff_min, param.cutoff_max))
        normalisation_layers.append(hist_norm)
    if param.whitening:
        mean_std_norm = MVNorm(
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
        import engine.training

        device_str = "gpu" if param.num_gpus > 0 else "cpu"
        engine.training.run(net_class, param, volume_loader, device_str)
    else:
        import engine.inference

        device_str = "gpu" if param.num_gpus > 0 else "cpu"
        engine.inference.run(net_class, param, volume_loader, device_str)
