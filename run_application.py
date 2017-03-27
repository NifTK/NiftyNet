import os
import sys
import parse_user_params
import util
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
        print "network: \"{}\" not implemented".format(name)
        raise NotImplementedError

if __name__ == "__main__":
    args = parse_user_params.run()
    if util.has_bad_inputs(args):
        sys.exit(-1)
    is_training = True if args.action == "train" else False
    device_str = "cpu" if (args.action=="train" and args.num_gpus>1) else "gpu"

    net_class = NetFactory.create(args.net_name)
    net = net_class(args.batch_size,
                    args.image_size,
                    args.label_size,
                    args.num_classes,
                    is_training = is_training,
                    device_str = device_str)
    if is_training:
        import training
        training.run(net, args)
    else:
        import inference
        inference.run(net, args)
