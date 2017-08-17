# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

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
import sys

import niftynet.utilities.util_common as util
import niftynet.utilities.user_parameters_parser as user_parameters_parser

from niftynet.engine.application_driver import ApplicationDriver

# if sys.version_info[0] >= 3:
#    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')
# else:
#    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)


def main():
    system_param, input_data_param = user_parameters_parser.run()
    if util.has_bad_inputs(system_param):
        return -1

    # print all parameters to txt file for future reference
    all_param = {}
    all_param.update(system_param)
    all_param.update(input_data_param)
    txt_file = 'settings_{}.txt'.format(system_param['APPLICATION'].action)
    txt_file = os.path.join(system_param['APPLICATION'].model_dir, txt_file)
    util.print_save_input_parameters(all_param, txt_file)

    app_driver = ApplicationDriver()
    app_driver.initialise_application(system_param, input_data_param)
    app_driver.run_application()
    return 0
    #if not (param.cuda_devices == '""'):
    #    os.environ["CUDA_VISIBLE_DEVICES"] = param.cuda_devices
    #    print("set CUDA_VISIBLE_DEVICES env to {}".format(param.cuda_devices))
    #net_class = NetFactory.create(param.net_name)

    ## expanding a few of the user input parameters
    #if param.spatial_rank == 3:
    #    spatial_padding = \
    #        ((param.volume_padding_size, param.volume_padding_size),
    #         (param.volume_padding_size, param.volume_padding_size),
    #         (param.volume_padding_size, param.volume_padding_size))
    #else:
    #    spatial_padding = \
    #        ((param.volume_padding_size, param.volume_padding_size),
    #         (param.volume_padding_size, param.volume_padding_size))
    #interp_order = (param.image_interp_order,
    #                param.label_interp_order,
    #                param.w_map_interp_order)

    ## read each line of csv files into an instance of Subject
    #csv_loader = CSVTable(csv_dict=csv_dict, allow_missing=True)

    ## define layers of volume-level normalisation
    #normalisation_layers = []
    #if param.normalisation:
    #    hist_norm = HistogramNormalisationLayer(
    #        models_filename=param.histogram_ref_file,
    #        binary_masking_func=BinaryMaskingLayer(
    #            type=param.mask_type,
    #            multimod_fusion=param.multimod_mask_type),
    #        norm_type=param.norm_type,
    #        cutoff=(param.cutoff_min, param.cutoff_max))
    #    normalisation_layers.append(hist_norm)
    #if param.whitening:
    #    mean_std_norm = MeanVarNormalisationLayer(
    #        binary_masking_func=BinaryMaskingLayer(
    #            type=param.mask_type,
    #            multimod_fusion=param.multimod_mask_type))
    #    normalisation_layers.append(mean_std_norm)

    ## define how to load image volumes
    #volume_loader = VolumeLoaderLayer(
    #    csv_loader,
    #    standardisor=normalisation_layers,
    #    is_training=(param.action == "train"),
    #    do_reorientation=param.reorientation,
    #    do_resampling=param.resampling,
    #    spatial_padding=spatial_padding,
    #    interp_order=interp_order)
    #print('found {} subjects'.format(len(volume_loader.subject_list)))

    #if param.action == "train":
    #    import niftynet.engine.training

    #    device_str = "gpu" if param.num_gpus > 0 else "cpu"
    #    niftynet.engine.training.run(net_class, param, volume_loader, device_str)
    #else:
    #    import niftynet.engine.inference

    #    device_str = "gpu" if param.num_gpus > 0 else "cpu"
    #    niftynet.engine.inference.run(net_class, param, volume_loader, device_str)
    #return 0

