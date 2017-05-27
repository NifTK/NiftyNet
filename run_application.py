# -*- coding: utf-8 -*-
import os
import sys

import utilities.misc as util
import utilities.parse_user_params as parse_user_params

sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)


class NetFactory(object):
    @staticmethod
    def create(name):
        if name == "highres3dnet":
            from layer.highres3dnet import HighRes3DNet
            return HighRes3DNet
        elif name == "toynet":
            from layer.toynet import ToyNet
            return ToyNet
        elif name == "3dunet":
            from layer.unet_3d import UNet_3D
            return UNet_3D
        elif name == "vnet":
            from layer.vnet import VNet
            return VNet
        elif name == "deepmedic":
            from layer.deepmedic import DeepMedic
            return DeepMedic
        elif name == "scalenet":
            from layer.scalenet import ScaleNet
            return ScaleNet
        else:
            print("network: \"{}\" not implemented".format(name))
            raise NotImplementedError


if __name__ == "__main__":
    args = parse_user_params.run()
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

        engine.training.run(net_class, args, device_str)
    else:
        import engine.inference

        engine.inference.run(net, args, device_str)
