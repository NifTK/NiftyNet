# -*- coding: utf-8 -*-
import os
import sys

import utilities.misc_common as util
import utilities.parse_user_params as parse_user_params

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)


class NetFactory(object):
    @staticmethod
    def create(name):
        if name == "highres3dnet":
            from network.highres3dnet import HighRes3DNet
            return HighRes3DNet
        elif name == "toynet":
            from network.toynet import ToyNet
            return ToyNet
        elif name == "3dunet":
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
        else:
            print("network: \"{}\" not implemented".format(name))
            raise NotImplementedError


if __name__ == "__main__":
    args, csv_dict = parse_user_params.run()
    if util.has_bad_inputs(args):
        sys.exit(-1)
    if not (args.cuda_devices == '""'):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
        print("set CUDA_VISIBLE_DEVICES env to {}".format(args.cuda_devices))
    is_training = True if args.action == "train" else False
    device_str = "gpu" if args.action == "train" and args.num_gpus > 0 else "cpu"

    net_class = NetFactory.create(args.net_name)
    if is_training:
        import engine.training

        engine.training.run(net_class, args, csv_dict, device_str)
    else:
        import engine.inference

        engine.inference.run(net_class, args, csv_dict, device_str)
