# Developing new networks

NiftyNet allows users create new network, and share the network via [the model
zoo](./model_zoo.html).  To fully utilise this feature, a customised network
should be prepared in the following steps:

## New network and module
   Create a new network file, e.g. `new_net.py` and place this inside a python
   module directory, e.g. `my_network_collection/` together with a new
   `__init__.py` file.

## Make the module loadable
   Make sure the new network module can be discovered by NiftyNet
   by doing **either** of the following:

   * Place `my_network_collection/` inside `$NIFTYNET_HOME/extensions/`, with
     `$NIFTYNET_HOME` defined by `home` in
     [`[global]`](./model_zoo.html#global-settings) setting.
   * Append the directory of `my_network_collection/` (i.e. **the directory
     where this folder is located**) to your `$PYTHONPATH`.


## Extend `BaseNet`
   Create a new Python class, e.g. `NewNet` in `new_net.py` by inheriting the
   `BaseNet` class from
   [`niftynet.network.base_net`](./niftynet.network.base_net.html).
   [`niftynet.network.toynet`](./niftynet.network.toynet.html), a minimal
   working example of a fully convolutional network, could be a starting point
   for `NewNet`.

   ```python
   class ToyNet(BaseNet):
    def __init__(self, num_classes, name='ToyNet'):

        super(ToyNet, self).__init__(
            num_classes=num_classes, acti_func=acti_func, name=name)

        # network specific property
        self.hidden_features = 10

    def layer_op(self, images, is_training):
        # create layer instances
        conv_1 = ConvolutionalLayer(self.hidden_features,
                                    kernel_size=3,
                                    name='conv_input')

        conv_2 = ConvolutionalLayer(self.num_classes,
                                    kernel_size=1,
                                    acti_func=None,
                                    name='conv_output')

        # apply layer instances
        flow = conv_1(images, is_training)
        flow = conv_2(flow, is_training)

        return flow
   ```

## Implement operations
   In `NewNet`, implement `__init__()` function for network property
   initialisations, and implement `layer_op()` for network connections.

   The network properties can be used to specify the number of channels, kernel
   dilation factors, as well as sub-network components of the network.

   An example of sub-networks composition is presented in
   [Simulator GAN](./niftynet.network.simulator_gan.html).

   The layer operation function `layer_op()` should specify how the input
   tensors are connected to network layers.  For basic building blocks, using
   the ones in [`niftynet/layer/`](./niftynet.layer.html) are recommended. as
   the layers are implemented in a modular design (convenient for parameter
   sharing) and can handle 2D, 2.5D and 3D cases in a unified manner whenever
   possible.

## Call `NewNet` from application
   Finally training the network could be done by specifying the newly
   implemented network in the command line argument

   ```bash
   --name my_network_collection.new_net.NewNet
   ```

   (`my_network_collection.new_net` refer to the `new_net.py` file, and `NewNet`
   is the class name to be imported from `new_net.py`)

   Command to load `NewNet` with segmentation application using [pip installed
   NiftyNet](./installation.html) is:
   ```
   net_segment train -c /path/to/customised_config \
                     --name my_network_collection.new_net.NewNet
   ```
   alternatively, using NiftyNet cloned from [source code
   repository](./installation.html):
   ```
   python net_segment.py train -c /path/to/customised_config \
                               --name my_network_collection.new_net.NewNet
   ```

   See also the configuration doc for [`name` parameter](./config_spec.html#name).

## Share the network and trained weights
   Please consider submitting the design to our [model zoo](./model_zoo.html).
   
   _See also: [Contributor guide/submitting model zoo entries](./contributing.html#submitting-model-zoo-entries)._

