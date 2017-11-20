# Configuration file
### Quick reference
- [Input data specifications](#input-data-source-section)
- [SYSTEM](#system)
- [NETWORK](#network)
- [TRAINING](#training)
- [INFERENCE](#inference)
- [Global settings](#global-settings)

## Overview
[This folder](../config) presents a few examples of configuration files for different
applications.

To run a NiftyNet [application](../niftynet/application) or a customised
application which implements [BaseApplication](../niftynet/application/base_application.py),
a configuration file needs to be provided, for example,
by creating a `user_configuration.ini` file and using the file via:
```bash
python net_segment.py train --conf user_configuration.ini
```

All files should have two sections:
- [`[SYSTEM]`](#system)
- [`[NETWORK]`](#network)

If the `train` is specified, then a training paramter section is required:
- [`[TRAINING]`](#training)

If the `inference` is specified, then an inference paramter section is required:
- [`[INFERENCE]`](#inference)

The section listed above are common hyperparameters for all applications.

Additionally, an application specific section is required for each application
(Please find further comments on creating customised parser [here](../niftynet/utilities/user_parameters_custom.py)):
- `[GAN]` for generative adversirial networks
- `[SEGMENTATION]` for segmentation networks
- `[REGRESSION]` for regression networks
- `[AUTOENCODER]` for autoencoder networks

The [user parameter parser](../niftynet/utilities/user_parameters_parser.py)
tries to match the section names listed above.
All other section names will be treated as:
- [`input data source specifications`](#input-data-source-section).

The following sections specify paramters available within each section.

## Input data source section
|Params.| Type |Example|Default|
|---|---|---|---|
|[path_to_search](#path_topath_to_search)|String|`path_to_search=my_data/fold_1`|Niftynet home folder|
|[filename_contain](#filename_contain)|String or string array|`filename_contain=foo, bar`|`''`|
|[filename_not_contain](#filename_not_contain)|String or string array|`filename_not_contain=foo`|`''`|
|[interp_order](#interp_order)|Integer|`interp_order=0`|`3`|
|[pixdim](#pixdim)|Float array|`pixdim=1.2, 1.2, 1.2`|`''`|
|[axcodes](#axcodes)|String array|`axcodes=L, P, S`|`''`|
|[spatial_window_size](#spatial_window_size)|Integer array|`spatial_window_size=64, 64, 64`|`''`|
This section will be used by [ImageReader](../niftynet/io/image_reader.py)
to generate a list of [input images objects](../niftynet/io/image_type.py).
For example:
```ini
[T1Image]
path_to_search = ./example_volumes/image_folder
filename_contain = ('T1', 'subject')
filename_not_contain = ('T1c', 'T2')
spatial_window_size = (128, 128, 1)
pixdim = (1.0, 1.0, 1.0)
axcodes=(A, R, S)
interp_order = 3
```
Specifies a set of images (currently supports NIfTI format via [NiBabel library](http://nipy.org/nibabel/nifti_images.html))
from `./example_volumes/image_folder`, with filnames contain both `T1` and
`subject`, but not contain `T1c` and `T2`. These images will be read into
memory and transformed into "A, R, S" orientation
(using [NiBabel](http://nipy.org/nibabel/reference/nibabel.orientations.html)).
The images will also be transformed to have voxel size `(1.0, 1.0, 1.0)`
with an interpolation order of `3`.

This input source can be used alone, as a `T1` MRI input to an application.
It can also be used along with other modalities, a multi-modality example
can be find at [here](../config/default_multimodal_segmentation.ini).

The following sections describe system parameters that can be specified in the configuration file.


## SYSTEM
|Params.| Type |Example|Default|
|---|---|---|---|
|[cuda_devices](#cuda_devices)|Integers set `CUDA_VISIBLE_DEVICES` | `cuda_devices=0,1,2`|`''`|
|[num_threads](#num_threads)|Positive integer|`num_threads=1`|`2`|
|[num_gpus](#num_gpus)| Integer|`num_gpus=4`|`1`|
|[model_dir](#model_dir)|String|`model_dir=/User/test_dir`|The directory of current configuration file|
|[dataset_split_file](#dataset_split_file)|String|`dataset_split_file=/User/my_test`|`./dataset_split_file.csv`|

###### `cuda_devices`
Sets the environment variable `CUDA_VISIBLE_DEVICES` variable,
e.g. `0,2,3` uses devices 0, 2, 3 will be visible; device 1 is masked.

###### `num_threads`
Sets number of preprocessing threads for training.

######  `num_gpus`
Sets number of training GPUs.
The value should be the number of available GPUs at most.
This option is ignored if there's no GPU device.

######  `model_dir`
Directory to save/load intermediate training models and logs.
Niftynet tries to interpret this parameter as an absolute system path or a path relative to the current command.
It's defaulting to the directory of the current configuration file if left blank.

######  `dataset_split_file`
File assigning subjects to training/validation/inference subsets.
If the string is a relative path, Niftynet interpret this as relative to `model_dir`.

## NETWORK
|Params.| Type |Example|Default|
|---|---|---|---|
|[name](#name)|String|`name=niftynet.network.toynet.ToyNet`|`''`|
|[activation_function](#activation_function)|String|`activation_function=prelu`|`relu`|
|[batch_size](#batch_size)|Integer|`batch_size=10`|`2`|
|[decay](#decay)|Non-negative float|`decay=1e-5`|`0.0`|
|[reg_type](#reg_type)|String|`reg_type=L1`|`L2`|
|[volume_padding_size](#volume_padding_size)|Integer array|`volume_padding_size=4, 4, 4`|`0,0,0`|
|[window_sampling](#window_sampling)|String|`window_sampling=uniform`|`uniform`|
|[queue_length](#queue_length)|Integer|`queue_length=10`|`5`|

######  `name`
A network class from [niftynet/network](../niftynet/network) or from user specified module string.
Niftynet tries to import this string as a module specification.
E.g. Setting it to `niftynet.network.toynet.ToyNet` will import the `ToyNet` class defined in `niftynet/network/toynet.py`
(The relevant module path must be valid Python path).
There are also [some shortcuts](../niftynet/engine/application_factory.py) for Niftynet's default network modules.

######  `activation_function`
Sets the type of activation of the network.
Available choices are listed in `SUPPORTED_OP` in [activation layer](../niftynet/layer/activation.py).
Depending on its implementation, the network might ignore this option .

######  `batch_size`
Sets number of image windows to be processed at each iteration.
When `num_gpus` is greater than 1, `batch_size` is used for each computing device.
That is, the effective inputs at each iteration become `batch_size` x `num_gpus`.

###### `reg_type`
Type of trainable parameter regularisation; currently the available choices are "L1" and "L2".
The loss will be added to `tf.GraphKeys.REGULARIZATION_LOSSES` collection.
This option will be ignored if [decay](#decay) is `0.0`.

###### `decay`
Strength of regularisation, to help prevent overfitting.

###### `volume_padding_size`
Number of values padded at image volume level.
The padding effect is equivalent to `numpy.pad` with:
```python
numpy.pad(input_volume,
          (volume_padding_size[0],
           volume_padding_size[1],
           volume_padding_size[2], 0, 0),
          mode='minimum')
```
For 2-D inputs, the third dimension of `volume_padding_size` should be set to `0`,
e.g. `volume_padding_size=M,N,0`.
`volume_padding_size=M` is a shortcut for 3-D inputs, equivalent to `volume_padding_size=M,M,M`.

###### `window_sampling`
Type of sampler used to generate image windows from each image volume:
- uniform: fixed size uniformly distributed,
- resize: resize image to the window size.

###### `queue_length`
Integer specifies window buffer size used when sampling image windows from image volumes.
Image window samplers fill the buffer and networks read the buffer.
Because the network reads [batch_size](#batch_size) windows at each iteration,
this value is set to at least `batch_size * 2.5` to allow for a possible randomised buffer.
i.e. queue_length is `max(queue_length, round(batch_size * 2.5))`.


##### Volume-normalisation
Intensity based volume normalisation can be configured using a combination of parameters described below:

(1) Setting `normalisation=True` enables the [histogram-based normalisation](../niftynet/utilities/histogram_standardisation.py).
The relelavant configuration parameters are:
> `histogram_ref_file`, `norm_type`, `cutoff`, `foreground_type`, `multimod_foreground_type`.

These parameters are ignored and histogram-based noramalisation is disabled if `normalisation=False`.

(2) Setting `whitening=True` enables the volume level normalisation computed by `(I - mean(I))/std(I)`.
The relelavant configuration parameters are:
> `cutoff`, `foreground_type`, `multimod_foreground_type`.

These parameters are ignored and histogram-based noramalisation is disabled if `whitening=False`.

More specifically:

|Params.| Type |Example|Default|
|---|---|---|---|
|[normalisation](#volume-normalisation)|Boolean|`normalisation=True`|`False`|
|[whitening](#volume-normalisation)|Boolean|`whitening=True`|`False`|
|[histogram_ref_file](#volume-normalisation)|String|`histogram_ref_file=./hist_ref.txt`| `''`|
|[norm_type](#volume-normalisation)|String|`norm_type=percentile`| `percentile`|
|[cutoff](#volume-normalisation)|Float array (two elements)|`cutoff=0.1, 0.9`|`0.01, 0.99`|
|[foreground_type](#volume-normalisation)|String|`foreground_type=ostu_plus`|`ostu_plus`|
|[multimod_foreground_type](#volume-normalisation)|String|`multimod_foreground_type=and`|`and`|


###### `normalisation`
Boolean indicates if an histogram standardisation should be applied to the data.

###### `whitening`
Boolean indicates if the loaded image should be whitened,
that is, given input image `I`, returns  `(I - mean(I))/std(I)`.

###### `histogram_ref_file`
Name of the file that contains the normalisation parameter if it has been trained before or where to save it.

###### `norm_type`
Type of histogram landmarks used in histogram-based normalisation (percentile or quartile).

###### `cutoff`
Inferior and superior cutoff in histogram-based normalisation.

###### `foreground_type`
To generate a foreground mask and the normalisation will be applied to foreground only.
Available choices:
> `otsu_plus`, `otsu_minus`, `thresh_plus`, `thresh_minus`.

###### `multimod_foreground_type`
Strategies applied to combine foreground masks of multiple modalities, can take one of the following:
* `or` union of the available masks,
* `and` intersection of the available masks,
* `all` masks computed from each modality independently.

## TRAINING
|Params.| Type |Example|Default|
|---|---|---|---|
|[optimiser](#optimiser)|String|`optimiser=momentum`|`adam`|
|[sampler_per_volume](#sample_per_volume)|Postive integer|`sampler_per_volume=5`|`1`|
|[lr](#lr)|Float|`lr=0.001`|`0.1`|
|[loss_type](#loss_type)|String|`loss_type=CrossEntropy`|`Dice`|
|[starting_iter](#starting_iter)|Non-negative integer|`starting_iter=0`| `0`|
|[save_every_n](#save_every_n)|Integer|`save_every_n=5`|`500`|
|[tensorboard_every_n](#tensorboard_every_n)|Integer|`tensorboard_every_n=5`|`20`|
|[max_iter](#max_iter)|Integer|`max_iter=1000`|`10000`|
|[max_checkpoint](#max_checkpoint)|Integer|`max_checkpoint=5`|`100`|


###### `optimiser`
Type of optimiser for computing graph gradients.

###### `sample_per_volume`
Set number of samples to take from each image volume.

###### `lr`
The learning rate for the optimiser.

###### `loss_type`
Type of loss function.

###### `starting_iter`
The iteration to resume training model. Setting `starting_iter=0` starts the network from random initialisations.

###### `save_every_n`
Frequency of saving the current training model saving. Setting to a non-positive value to disable the saving schedule.
(A final model will always be saved when quitting the training loop.)

###### `tensorboard_every_n`
Frequency of evaluating graph elements and write to tensorboard.
Setting to a non-positive value to disable the tensorboard writing schedule.

###### `max_iter`
Maximum number of training iterations. The value is total number of iterations counting from 0.
This means when training from [`starting_iter`](#starting_iter) N,
the remaining number of iterations to run is `N - max_iter`.

###### `max_checkpoint`
Maximum number of recent checkpoints to keep.

##### Validation during training

|Params.| Type |Example|Default|
|---|---|---|---|
|[validation_every_n](#validation_every_n)|
|[validation_max_iter](#validation_max_iter)|
|[exclude_fraction_for_validation](#exclude_fraction_for_validation)|
|[exclude_fraction_for_inference](#exclude_fraction_for_inference)

###### `validation_every_n`

###### `validation_max_iter`

###### `exclude_fraction_for_validation`

###### `exclude_fraction_for_inference`

##### Data augmentation during training

|Params.| Type |Example|Default|
|---|---|---|---|
|[rotation_angle](#rotation_angle)|Float array|`rotation_angle=-10.0,10.0`|`''`|
|[scaling_percentage](#scaling_percentage)|Float array|`scaling_percentage=0.8,1.2`|`''`|
|[random_flipping_axes](#random_flipping_axes)|Integer array|`random_flipping_axes=1,2`|`-1`|

###### `rotation_angle`
Float array, indicates a random rotation operation should be applied to the volumes
(This can be slow depending on the input volume dimensionality).

###### `scaling_percentage`
Float array indicates a random spatial scaling should be applied
(This can be slow depending on the input volume dimensionality).

###### `random_flipping_axes`
The axes which can be flipped to augment the data.
Supply as comma-separated values within single quotes, e.g. '0,1'.
Note that these are 0-indexed, so choose some combination of 0, 1.


## INFERENCE
|Params.| Type |Example|Default|
|---|---|---|---|
|[spatial_window_size](#spatial_window_size)|
|[border](#border)|
|[inference_iter](#inference_iter)|Integer|`inference_iter=1000`|`-1`|
|[save_seg_dir](#save_seg_dir)|String|`save_seg_dir=output/test`| A new directory named `output` within [`model_dir`](#model_dir)|
|[output_interp_order](#output_interp_order)|Non-negative integer|`output_interp_order=0`|`0`|

###### `spatial_window_size`
a tuple of integers indicating the size of input window
at inference time, this overrides the `spatial_window_size` parameter in the input
source sections.

###### `border`
a tuple of integers specifying a border size used to crop (along both sides of each
dimension) the network output image window. E.g., `3, 3, 3` will crop a
`64x64x64` window to size `58x58x58`.

###### `inference_iter`
Integer specifies the trained model to be used for inference.
`-1` or unspecified indicating to use the latest available trained model in `model_dir`.

###### `save_seg_dir`
Prediction directory name

###### `output_interp_order`
Interpolation order of the network output.




## Global-settings

The global NiftyNet configuration is read from `~/.niftynet/config.ini`.
When NiftyNet is run, it will attempt to load this file for the global configuration.
* If it does not exist, NiftyNet will create a default one.
* If it exists but cannot be read (e.g., due to incorrect formatting):
- NiftyNet will back it up with a timestamp (for instance `~/.niftynet/config-backup-2017-10-16-10-50-58-abc.ini` - `abc` being a random string) and,
- Create a default one.
* Otherwise NiftyNet will read the global configuration from this file.

Currently the minimal version of this file will look like the following:
```ini
[global]
home = ~/niftynet
```

The `home` key specifies the root folder (referred to as `$NIFTYNET_HOME` from this point onwards) to be used by NiftyNet for storing and locating user data such as downloaded models, and new networks implemented by the user.
This setting is configurable, and upon successfully loading this file NiftyNet will attempt to create the specified folder, if it does not already exist.

On first run, NiftyNet will also attempt to create the NiftyNet extension module hierarchy (`niftynetext.*`), that allows for the discovery of user-defined networks.
This hierarchy consists of the following:

* `$NIFTYNET_HOME/niftynetext/` (folder)
* `$NIFTYNET_HOME/niftynetext/__init__.py` (file)
* `$NIFTYNET_HOME/niftynetext/network/` (folder)
* `$NIFTYNET_HOME/niftynetext/network/__init__.py` (file)

Alternatively this hierarchy can be created by the user before running NiftyNet for the first time, e.g. for [defining new networks][new-network].

[new-network]: ../niftynet/network/README.md






