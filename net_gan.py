# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys

import niftynet.utilities.misc_common as util
import niftynet.utilities.parse_user_params_gan as parse_user_params
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
        if name == "simulator_gan":
            from niftynet.network.simulator_gan import SimulatorGAN
            return SimulatorGAN
        else:
            print("network: \"{}\" not implemented".format(name))
            raise NotImplementedError


def main():
    param, csv_dict = parse_user_params.run()
    if util.has_bad_inputs(param):
        return -1
    else:
        util.print_save_input_parameters(param, txt_file=os.path.join(param.model_dir,
                                                                      'settings_' + param.action + '.txt'))

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
                    param.conditioning_interp_order)

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
if __name__ == "__main__":
  sys.exit(main())
