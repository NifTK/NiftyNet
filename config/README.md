# Configuration file
*[This folder](../config) presents a few examples of configuration files for NiftyNet
[applications](#../niftynet/application/).*

*This readme file describes commands and configurations supported by NiftyNet.*

##### Quick reference
- [Input data specifications](#input-data-source-section)
- [SYSTEM](#system)
- [NETWORK](#network)
  * [Volume-normalisation](#volume-normalisation)
- [TRAINING](#training)
  * [Validation during training](#validation-during-training)
  * [Data augmentation during training](#data-augmentation-during-training)
- [INFERENCE](#inference)
- [Global settings](#global-settings)

## Overview
In general, a NiftyNet workflow can be fully specified by a NiftyNet application and a configuration file.
The command to run the workflow is:
```bash
# command to run from git-cloned NiftyNet source code folder
python net_run.py [train|inference] -c <path_to/config.ini> -a <application>
```
or:
```bash
# command to run using pip-installed NiftyNet
net_run [train|inference] -c <path_to/config.ini> -a <application>
```
`net_run` is the entry point of NiftyNet, followed by an action argument of either `train`
or `inference`. `train` indicates updating the underlying network model using provided data.
`inference` indicates loading existing network model and generating responses according to data provided.

#### The `<application>` argument
`<application>` should be specified in the form of `user.path.python.module.MyApplication`,
NiftyNet will try to import the class named `MyApplication` implemented in `user/path/python/module.py`.

A few applications are already included in NiftyNet, and can be passed as an argument of `-a`.
These include:

|Argument|Workflow|File|
|---|---|---|
|`niftynet.application.segmentation_application.SegmentationApplication`|image segmentation |[segmentation_application.py](../niftynet/application/segmentation_application.py)|
|`niftynet.application.regression_application.RegressionApplication`|image regression|[regression_application.py](../niftynet/application/regression_application.py)|
|`niftynet.application.autoencoder_application.AutoencoderApplication`|autoencoder|[autoencoder_application.py](../niftynet/application/autoencoder_application.py)|
|`niftynet.application.gan_application.GANApplication`|generative adversarial network|[gan_application.py](../niftynet/application/gan_application.py)|

Shortcuts are created for these application (full specification can be found here: [`SUPPORTED_APP`](../niftynet/engine/application_factory.py#L19)):

|Shortcut| Equivalent partial command|
|---|---|
|`net_segment`|`net_run -a niftynet.application.segmentation_application.SegmentationApplication`|
|`net_regress`|`net_run -a niftynet.application.regression_application.RegressionApplication`|
|`net_autoencoder`|`net_run -a niftynet.application.autoencoder_application.AutoencoderApplication`|
|`net_gan`|`net_run -a niftynet.application.gan_application.GANApplication`|

#### Overriding the arguments
In the case of quickly adjusting only a few options in the configuration file, creating a separate file is sometimes tedious.

To make it more accessible, `net_run` command also accepts parameters specification in the form of `--<name> <value>` or `--<name>=<value>`.
When these are used, `value` will override the corresponding value of `name` defined both by system default and configuration file.

The following sections describes content of a configuration file `<path_to/config.ini>`.

## Required and optional configuration sections
The configuration file currently adopts the INI file format, and is parsed by
[`configparser`](https://docs.python.org/3/library/configparser.html).
The file consists of multiple sections of `name=value` elements.

All files should have two sections:
- [`[SYSTEM]`](#system)
- [`[NETWORK]`](#network)

If  `train` action is specified, then a [`[TRAINING]`](#training) section is required.

If  `inference` action is specified, then an [`[INFERENCE]`](#inference) section is required.

Additionally, an application specific section is required for each application
(Please find further comments on creating customised parser [here](../niftynet/utilities/user_parameters_custom.py)):
- `[GAN]` for generative adversarial networks
- `[SEGMENTATION]` for segmentation networks
- `[REGRESSION]` for regression networks
- `[AUTOENCODER]` for autoencoder networks

The [user parameter parser](../niftynet/utilities/user_parameters_parser.py)
tries to match the section names listed above.
All other section names will be treated as [`input data source specifications`](#input-data-source-section).

The following sections specify parameters (`<name> = <value>` pairs) available within each section.

## Input data source section
|Params.| Type |Example|Default|
|---|---|---|---|
|[csv_file](#csv_file)|String|`csv_file=file_list.csv`|`''`|
|[path_to_search](#path_to_search)|String|`path_to_search=my_data/fold_1`|NiftyNet home folder|
|[filename_contains](#filename_contains)|String or string array|`filename_contains=foo, bar`|`''`|
|[filename_not_contains](#filename_not_contains)|String or string array|`filename_not_contains=foo`|`''`|
|[interp_order](#interp_order)|Integer|`interp_order=0`|`3`|
|[pixdim](#pixdim)|Float array|`pixdim=1.2, 1.2, 1.2`|`''`|
|[axcodes](#axcodes)|String array|`axcodes=L, P, S`|`''`|
|[spatial_window_size](#spatial_window_size)|Integer array|`spatial_window_size=64, 64, 64`|`''`|

###### `csv_file`
A file path to a list of input images.
If the file exists, input image name list will be loaded from the file;
the filename based input image search will be disabled;
[path_to_search](#path_to_search), [filename_contains](#filename_contains), and [filename_not_contains](#filename_not_contains)
will be ignored.
If this parameter is left blank or the file does not exist,
input image search will be enabled, and the matched filenames will be written to this file path.

###### `path_to_search`
Single or multiple folders to search for input images.

###### `filename_contains`
Keywords used to match filenames.
The matched keywords will be removed, and the remaining part is used as
subject name (for loading corresponding images across modalities).

###### `filename_not_contains`
Keywords used to exclude filenames.
The filenames with these keywords will not be used as input.

###### `interp_order`
Interpolation order of the input data.

###### `pixdim`
If specified, the input volume will be resampled to the voxel sizes
before fed into the network.

###### `axcodes`
If specified, the input volume will be reoriented to the axes codes
before fed into the network.

###### `spatial_window_size`
Array of three integers specifies the input window size.
Setting it to single slice, e.g., `spatial_window_size=64, 64, 1`, yields a 2-D slice window.

This section will be used by [ImageReader](../niftynet/io/image_reader.py)
to generate a list of [input images objects](../niftynet/io/image_type.py).
For example:
```ini
[T1Image]
path_to_search = ./example_volumes/image_folder
filename_contains = T1, subject
filename_not_contains = T1c, T2
spatial_window_size = 128, 128, 1
pixdim = 1.0, 1.0, 1.0
axcodes = A, R, S
interp_order = 3
```
Specifies a set of images
(currently supports NIfTI format via [NiBabel library](http://nipy.org/nibabel/nifti_images.html))
from `./example_volumes/image_folder`, with filenames contain both `T1` and
`subject`, but not contain `T1c` and `T2`. These images will be read into
memory and transformed into "A, R, S" orientation
(using [NiBabel](http://nipy.org/nibabel/reference/nibabel.orientations.html)).
The images will also be transformed to have voxel size `(1.0, 1.0, 1.0)`
with an interpolation order of `3`.

A CSV file with the matched filenames and extracted subject names will be generated
to `T1Image.csv` in [`model_dir`](#model_dir)
(by default; the CSV file location can be specified by setting [csv_file](#csv_file)).
To exclude particular images,
the [csv_file](#csv_file) can be edited manually.

This input source can be used alone, as a `T1` MRI input to an application.
It can also be used along with other modalities, a multi-modality example
can be find at [here](../config/default_multimodal_segmentation.ini).

The following sections describe system parameters that can be specified in the configuration file.


## SYSTEM
|Params.|Type|Example|Default|
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
NiftyNet tries to interpret this parameter as an absolute system path or a path relative to the current command.
It's defaulting to the directory of the current configuration file if left blank.

######  `dataset_split_file`
File assigning subjects to training/validation/inference subsets.
If the string is a relative path, NiftyNet interpret this as relative to `model_dir`.


## NETWORK
|Params.|Type|Example|Default|
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
NiftyNet tries to import this string as a module specification.
E.g. Setting it to `niftynet.network.toynet.ToyNet` will import the `ToyNet` class defined in `niftynet/network/toynet.py`
(The relevant module path must be a valid Python path).
There are also some shortcuts ([`SUPPORTED_NETWORK`](../niftynet/engine/application_factory.py#L30)) for NiftyNet's default network modules.

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
The same amount of padding will be removed when before writing the output volume.

###### `window_sampling`
Type of sampler used to generate image windows from each image volume:
- uniform: fixed size uniformly distributed,
- weighted: fixed size where the likelihood of sampling a voxel is proportional to the cumulative intensity histogram,
- balanced: fixed size where each label has the same probability of being sampled,
- resize: resize image to the window size.
For _weighted_ and _balanced_, an image reader named `sampler` is required to generate windows.

###### `queue_length`
Integer specifies window buffer size used when sampling image windows from image volumes.
Image window samplers fill the buffer and networks read the buffer.
Because the network reads [batch_size](#batch_size) windows at each iteration,
this value is set to at least `batch_size * 2.5` to allow for a possible randomised buffer,
i.e. `max(queue_length, round(batch_size * 2.5))`.


##### Volume-normalisation
Intensity based volume normalisation can be configured using a combination of parameters described below:

(1) Setting `normalisation=True` enables the [histogram-based normalisation](../niftynet/utilities/histogram_standardisation.py).
The relevant configuration parameters are:
> `histogram_ref_file`, `norm_type`, `cutoff`, `normalise_foreground_only`, `foreground_type`, `multimod_foreground_type`.

These parameters are ignored and histogram-based normalisation is disabled if `normalisation=False`.

(2) Setting `whitening=True` enables the volume level normalisation computed by `(I - mean(I))/std(I)`.
The relevant configuration parameters are:
> `normalise_foreground_only`, `foreground_type`, `multimod_foreground_type`.

These parameters are ignored and whitening is disabled if `whitening=False`.

More specifically:

|Params.|Type|Example|Default|
|---|---|---|---|
|[normalisation](#normalisation)|Boolean|`normalisation=True`|`False`|
|[whitening](#whitening)|Boolean|`whitening=True`|`False`|
|[histogram_ref_file](#histogram_ref_file)|String|`histogram_ref_file=./hist_ref.txt`| `''`|
|[norm_type](#norm_type)|String|`norm_type=percentile`| `percentile`|
|[cutoff](#cutoff)|Float array (two elements)|`cutoff=0.1, 0.9`|`0.01, 0.99`|
|[normalise_foreground_only](#normalise_foreground_only)|Boolean|`normalise_foreground_only=True`|`False`|
|[foreground_type](#foreground_type)|String|`foreground_type=otsu_plus`|`otsu_plus`|
|[multimod_foreground_type](#multimod_foreground_type)|String|`multimod_foreground_type=and`|`and`|

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

###### `normalise_foreground_only`
Boolean indicates if a mask should be computed based on `foreground_type` and `multimod_foreground_type`.
If this parameter is set to `True`, all normalisation steps will be applied to the generated foreground
regions only.

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
|[sample_per_volume](#sample_per_volume)|Positive integer|`sample_per_volume=5`|`1`|
|[lr](#lr)|Float|`lr=0.001`|`0.1`|
|[loss_type](#loss_type)|String|`loss_type=CrossEntropy`|`Dice`|
|[starting_iter](#starting_iter)|Non-negative integer|`starting_iter=0`| `0`|
|[save_every_n](#save_every_n)|Integer|`save_every_n=5`|`500`|
|[tensorboard_every_n](#tensorboard_every_n)|Integer|`tensorboard_every_n=5`|`20`|
|[max_iter](#max_iter)|Integer|`max_iter=1000`|`10000`|
|[max_checkpoints](#max_checkpoints)|Integer|`max_checkpoints=5`|`100`|

###### `optimiser`
Type of optimiser for computing graph gradients.
Current available options are defined here: [`SUPPORTED_OPTIMIZERS`](../niftynet/engine/application_factory.py#L106).

###### `sample_per_volume`
Set number of samples to take from each image volume.

###### `lr`
The learning rate for the optimiser.

###### `loss_type`
Type of loss function.
Please see the relevant loss function layer for choices available:
- [Segmentation](../niftynet/layer/loss_segmentation.py),
- [Regression](../niftynet/layer/loss_regression.py),
- [Autoencoder](../niftynet/layer/loss_autoencoder.py),
- [GAN](../niftynet/layer/loss_gan.py).

The corresponding loss function type names are defined in the
[`ApplicationFactory`](../niftynet/engine/application_factory.py#L67)


###### `starting_iter`
The iteration to resume training model.
Setting `starting_iter=0` starts the network from random initialisations.

###### `save_every_n`
Frequency of saving the current training model saving.
Setting to a `0` to disable the saving schedule.
(A final model will always be saved when quitting the training loop.)

###### `tensorboard_every_n`
Frequency of evaluating graph elements and write to tensorboard.
Setting to `0` to disable the tensorboard writing schedule.

###### `max_iter`
Maximum number of training iterations.
The value is total number of iterations counting from 0.
This means when training from [`starting_iter`](#starting_iter) N,
the remaining number of iterations to run is `N - max_iter`.

###### `max_checkpoints`
Maximum number of recent checkpoints to keep.

##### Validation during training
Setting [`validation_every_n`](#validation_every_n) to a positive integer enables validation loops during training.
When validation is enabled, images list (defined by [input specifications](#input-data-source-section))
will be treated as the whole dataset, and partitioned into subsets of training, validation, and inference
according to [exclude_fraction_for_validation](#exclude_fraction_for_validation) and
[exclude_fraction_for_inference](#exclude_fraction_for_inference).

A CSV table randomly mapping each file name to one of the stages `{'Training', 'Validation', 'Inference'}` will be generated and written to
[dataset_split_file](#dataset_split_file). This file will be created at the beginning of training (`starting_iter=0`) and
only if the file does not exist.

- If a new random partition is required, please remove the existing [dataset_split_file](#dataset_split_file).

- If no partition is required, please remove any existing [dataset_split_file](#dataset_split_file),
and make sure both [exclude_fraction_for_validation](#exclude_fraction_for_validation)
and [exclude_fraction_for_inference](#exclude_fraction_for_inference) are `0`.

To exclude particular subjects or adjust the randomly generated partition,
the [dataset_split_file](#dataset_split_file) can be edited manually.
Please note duplicated rows are not removed. For example, if the content of [dataset_split_file](#dataset_split_file) is as follows:
```csv
1040,Training
1071,Inference
1071,Inference
1065,Training
1065,Training
1065,Validation
```
Each row will be treated as an independent subject. This means:
>subject `1065` will be used in both `Training` and `Validation` stages, and it'll be sampled more frequently than subject `1040` during training;
>subject `1071` will be used in `Inference` twice, the output of the second inference will overwrite the first.

Note that at each validation iteration, input will be sampled from the set of validation data,
and the network parameters will remain unchanged.  The `is_training` parameter of the network
is set to `True` during validation, as a result layers with different behaviours in training and inference
(such as dropout and batch normalisation) uses the training behaviour.

During inference, if a [dataset_split_file](#dataset_split_file) is available, only image files in
the `Inference` phase will be used,
otherwise inference will process all image files defined by [input specifications](#input-data-source-section).

|Params.| Type |Example|Default|
|---|---|---|---|
|[validation_every_n](#validation_every_n)| Integer|`validation_every_n=10`|`-1`|
|[validation_max_iter](#validation_max_iter)|Integer|`validation_max_iter=5`|`1`|
|[exclude_fraction_for_validation](#exclude_fraction_for_validation)|Float|`exclude_fraction_for_validation=0.2`|`0.0`|
|[exclude_fraction_for_inference](#exclude_fraction_for_inference)|Float|`exclude_fraction_for_inference=0.1`|`0.0`|

###### `validation_every_n`
Run validation iterations after every N training iterations.
Setting to `0` disables the validation.

###### `validation_max_iter`
Number of validation iterations to run.
This parameter is ignored if `validation_every_n` is not a positive integer.

###### `exclude_fraction_for_validation`
Fraction of dataset to use for validation.
Value should be in `[0, 1]`.

###### `exclude_fraction_for_inference`
Fraction of dataset to use for inference.
Value should be in `[0, 1]`.

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
Many networks are fully convolutional (without fully connected layers) and
the resolution of the output volume can be different from the input image.
That is, given input of an `NxNxN` voxel volume, the network generates
a `DxDxD`-voxel output, where `0 < D < N`.

This configuration section is design for such a process of sampling `NxNxN` windows
from image volumes, and aggregating the network-generated `DxDxD` windows to output
volumes.

In terms of sampling by a sliding window, the sampling step size should be `D/2` in each
spatial dimension.  However automatically inferring `D` as a function of network architecture and `N`
is not implemented at the moment. Therefore, NiftyNet requires a [`border`](#border) to describe the
spatial window size changes. `border` should be at least `floor((N - D) / 2)`.

If the network is designed such that `N==D` is always true, `border` should be `0` (default value).

Note that the above implementation generalises to
`NxMxP`-voxel windows and `BxCxD`-voxel window outputs.
For a 2-D slice, e.g, `Nx1xM`, the second dimension of `border` should be `0`.

|Params.| Type |Example|Default|
|---|---|---|---|
|[spatial_window_size](#spatial_window_size)|Integer array| `spatial_window_size=64,64,64`|`''`|
|[border](#border)|Integer array|`border=5,5,5`|`0, 0, 0`|
|[inference_iter](#inference_iter)|Integer|`inference_iter=1000`|`-1`|
|[save_seg_dir](#save_seg_dir)|String|`save_seg_dir=output/test`| `output`|
|[output_interp_order](#output_interp_order)|Non-negative integer|`output_interp_order=0`|`0`|
|[dataset_to_infer](#dataset_to_infer) :: `Training` or `Validation` or `Inference` :: `dataset_to_infer=Training` :: `''``

###### `spatial_window_size`
Array of integers indicating the size of input window.
By default, the window size at inference time is the same as the [input source specification](#input-data-source-section).
If this parameter is specified, it overrides the `spatial_window_size` parameter in input
source sections.

###### `border`
a tuple of integers specifying a border size used to crop (along both sides of each
dimension) the network output image window. E.g., `3, 3, 3` will crop a
`64x64x64` window to size `58x58x58`.

###### `inference_iter`
Integer specifies the trained model to be used for inference.
`-1` or unspecified indicating to use the latest available trained model in `model_dir`.

###### `save_seg_dir`
Prediction directory name. If it's a relative path, it is set to be relative to [`model_dir`](#model_dir).

###### `output_interp_order`
Interpolation order of the network outputs.

###### `dataset_to_infer`
String Specifies which dataset ('Training', 'Validation', 'Inference') to compute inference for.
By default 'Inference' dataset is used.

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

On first run, NiftyNet will also attempt to create the NiftyNet extension module hierarchy (`extensions.*`), that allows for the discovery of user-defined networks.
This hierarchy consists of the following:

* `$NIFTYNET_HOME/extensions/` (folder)
* `$NIFTYNET_HOME/extensions/__init__.py` (file)
* `$NIFTYNET_HOME/extensions/network/` (folder)
* `$NIFTYNET_HOME/extensions/network/__init__.py` (file)

Alternatively this hierarchy can be created by the user before running NiftyNet for the first time, e.g. for [defining new networks][new-network].

[new-network]: ../niftynet/network/README.md
