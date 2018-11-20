# Configuration file
*[The config folder](https://github.com/NifTK/NiftyNet/tree/dev/config)
presents a few examples of configuration files for NiftyNet
[applications](./niftynet.application.html).*

*This page describes commands and configurations supported by NiftyNet.*

##### Quick reference
- [`[INPUT DATA SPECIFICATIONS]`](#input-data-source-section)
- [`[SYSTEM]`](#system)
- [`[NETWORK]`](#network)
  * [`[Volume-normalisation]`](#volume-normalisation)
- [`[TRAINING]`](#training)
  * [`[Validation during training]`](#validation-during-training)
  * [`[Data augmentation during training]`](#data-augmentation-during-training)
- [`[INFERENCE]`](#inference)
- [`[EVALUATION]`](#evaluation)

## Overview
In general, a NiftyNet workflow can be fully specified by a NiftyNet application and a configuration file.
The command to run the workflow is:
```bash
# command to run from git-cloned NiftyNet source code folder
python net_run.py [train|inference|evaluation] -c <path_to/config.ini> -a <application>
```
or:
```bash
# command to run using pip-installed NiftyNet
net_run [train|inference|evaluation] -c <path_to/config.ini> -a <application>
```
`net_run` is the entry point of NiftyNet, followed by an action argument of either `train`
or `inference`:
- `train` indicates updating the underlying network model using provided data.
- `inference` indicates loading existing network model and generating responses according to data provided.

#### The `<application>` argument
`<application>` should be specified in the form of `user.path.python.module.MyApplication`,
NiftyNet will try to import the class named `MyApplication` implemented in `user/path/python/module.py`.

A few applications are already included in NiftyNet, and can be passed as an
argument of `-a`.  Aliases are also created for these application (full
specification can be found here:
[`SUPPORTED_APP`](./niftynet.engine.application_factory.html)):
The commands include:

- [image segmentation](./niftynet.application.segmentation_application.html)
```bash
# command
net_run -a niftynet.application.segmentation_application.SegmentationApplication -c ...
# alias:
net_segment -c ...
```

- [image regression](./niftynet.application.regression_application.html)
```bash
# command
net_run -a niftynet.application.regression_application.RegressionApplication -c ...
# alias:
net_regress -c ...
```

- [autoencoder](./niftynet.application.autoencoder_application.html)
```bash
# command
net_run -a niftynet.application.autoencoder_application.AutoencoderApplication -c ...
# alias:
net_autoencoder -c ...
```

- [generative adversarial network](./niftynet.application.gan_application.html)
```bash
# command
net_run -a niftynet.application.gan_application.GANApplication -c ...
# alias:
net_gan -c ...
```

#### Overriding the arguments
In the case of quickly adjusting only a few options in the configuration file,
creating a separate file is sometimes tedious.

To make it more accessible, `net_run` command also accepts parameters
specification in the form of `--<name> <value>` or `--<name>=<value>`.  When
these are used, `value` will override the corresponding value of `name` defined
both by system default and configuration file.

The following sections describes content of a configuration file
`<path_to/config.ini>`.

## Configuration sections
The configuration file currently adopts the INI file format, and is parsed by
[`configparser`](https://docs.python.org/3/library/configparser.html).
The file consists of multiple sections of `name=value` elements.

All files should have two sections:
- [`[SYSTEM]`](#system)
- [`[NETWORK]`](#network)

If  `train` action is specified, then a [`[TRAINING]`](#training) section is required.

If  `inference` action is specified, then an [`[INFERENCE]`](#inference) section is required.

Additionally, an application specific section is required for each application
(Please find further comments on [creating customised parser here](https://github.com/NifTK/NiftyNet/blob/dev/niftynet/utilities/user_parameters_custom.py)):
- `[GAN]` for generative adversarial networks
- `[SEGMENTATION]` for segmentation networks
- `[REGRESSION]` for regression networks
- `[AUTOENCODER]` for autoencoder networks

The [user parameter parser](../niftynet/utilities/user_parameters_parser.py)
tries to match the section names listed above.  All other section names will be
treated as [`input data source specifications`](#input-data-source-section).

The following sections specify parameters (`<name> = <value>` pairs) available
within each section.

### Input data source section


 Name | Type | Example | Default
 ---- | ---- | ------- | -------
[csv_file](#csv-file) | `string` | `csv_file=file_list.csv` | `''`
[path_to_search](#path-to-search) | `string` | `path_to_search=my_data/fold_1` | NiftyNet home folder
[filename_contains](#filename-contains) | `string` or `string array` | `filename_contains=foo, bar` | `''`
[filename_not_contains](#filename-not-contains) | `string` or `string array` | `filename_not_contains=foo` | `''`
[filename_removefromid](#filename-removefromid) | `string` | `filename_removefromid=bar` | `''`
[interp_order](#interp-order) | `integer` | `interp_order=0` | `3`
[pixdim](#pixdim) | `float array` | `pixdim=1.2, 1.2, 1.2` | `''`
[axcodes](#axcodes) | `string array` | `axcodes=L, P, S` | `''`
[spatial_window_size](#spatial-window-size) | `integer array` | `spatial_window_size=64, 64, 64` | `''`
[loader](#loader) | `string` | `loader=simpleitk` | `None`

###### `csv_file`
A file path to a list of input images.  If the file exists, input image name
list will be loaded from the file; the filename based input image search will
be disabled; [path_to_search](#path-to-search),
[filename_contains](#filename-contains),
 [filename_not_contains](#filename-not-contains),
 and [filename_removefromid](#filename-removefromid) will be ignored.  If this
parameter is left blank or the file does not exist, input image search will be
enabled, and the matched filenames will be written to this file path.

###### `path_to_search`
Single or multiple folders to search for input images.

See also: [input filename matching guide](./filename_matching.html)


###### `filename_contains`
Keywords used to match filenames.
The matched keywords will be removed, and the remaining part is used as
subject name (for loading corresponding images across modalities).

See also: [input filename matching guide](./filename_matching.html)

###### `filename_not_contains`
Keywords used to exclude filenames.
The filenames with these keywords will not be used as input.

See also: [input filename matching guide](./filename_matching.html)

###### `filename_removefromid`
Regular expression for extracting subject id from filename,
matched pattern will be removed from the file names to form the subject id.

See also: [input filename matching guide](./filename_matching.html)

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

See also: [Patch-base analysis guide](./window_sizes.html)

###### `loader`
Specify the loader to be used to load the files in the input section.
Some loaders require additional Python packages.
Default value `None` indicates trying all available loaders.

This section will be used by [ImageReader](./niftynet.io.image_reader.html)
to generate a list of [input images objects](./niftynet.io.image_type.html).
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

A CSV file with the matched filenames and extracted subject names will be
generated to `T1Image.csv` in [`model_dir`](#model-dir) (by default; the CSV
file location can be specified by setting [csv_file](#csv-file)).  To exclude
particular images, the [csv_file](#csv-file) can be edited manually.

This input source can be used alone, as a `T1` MRI input to an application.
It can also be used along with other modalities, a multi-modality example
can be find [here](https://github.com/NifTK/NiftyNet/blob/dev/config/default_multimodal_segmentation.ini).

The following sections describe system parameters that can be specified in the configuration file.


### SYSTEM

 Name | Type | Example | Default
 ---- | ---- | ------- | -------
[cuda_devices](#cuda-devices) | `integer array` | `cuda_devices=0,1,2` | `''`
[num_threads](#num-threads) | `positive integer` | `num_threads=1` | `2`
[num_gpus](#num-gpus) | `integer` | `num_gpus=4` | `1`
[model_dir](#model-dir) | `string` | `model_dir=/User/test_dir` | The directory of the config. file
[dataset_split_file](#dataset-split-file) | `string` | `dataset_split_file=/User/my_test` | `./dataset_split_file.csv`
[event_handler](#event-handler) | `string` or a list of `string`s | `event_handler=model_restorer` | `model_saver, model_restorer, sampler_threading, apply_gradients, output_interpreter, console_logger, tensorboard_logger`

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
Directory to save/load intermediate training models and logs.  NiftyNet tries
to interpret this parameter as an absolute system path or a path relative to
the current command.  It's defaulting to the directory of the current
configuration file if left blank.

######  `dataset_split_file`
File assigning subjects to training/validation/inference subsets.
If the string is a relative path, NiftyNet interpret this as relative to `model_dir`.

######  `event_handler`
Event handler functions registered to these signals will be called by the
engine, along with NiftyNet application properties and iteration messages as
function parameters. See [Signals and event handlers](extending_event_handler.html) for more details.


### NETWORK

 Name | Type | Example | Default
 ---- | ---- | ------- | -------
[name](#name) | `string` | `name=niftynet.network.toynet.ToyNet` | `''`
[activation_function](#activation-function) | `string` | `activation_function=prelu` | `relu`
[batch_size](#batch-size) | `integer` | `batch_size=10` | `2`
[smaller_final_batch_mode](#smaller-final-batch-mode) | `string` | | `pad`
[decay](#decay) | `non-negative float` | `decay=1e-5` | `0.0`
[reg_type](#reg-type) | `string` | `reg_type=L1` | `L2`
[volume_padding_size](#volume-padding-size) | `integer array` | `volume_padding_size=4, 4, 4` | `0,0,0`
[volume_padding_mode](#volume-padding-mode) | `string` | `volume_padding_mode=symmetric` | `minimum`
[window_sampling](#window-sampling) | `string` | `window_sampling=uniform` | `uniform`
[queue_length](#queue-length) | `integer` | `queue_length=10` | `5`
[keep_prob](#keep-prob) | `non-negative float` | `keep_prob=0.2` | `1.0`

######  `name`
A network class from [niftynet/network](./niftynet.network.html) or from user
specified module string.  NiftyNet tries to import this string as a module
specification.  For example, setting it to `niftynet.network.toynet.ToyNet`
will import the `ToyNet` class defined in
[`niftynet/network/toynet.py`](./niftynet.network.toynet.html)
(The relevant module path must be a valid Python path).
There are also some shortcuts
([`SUPPORTED_NETWORK`](./niftynet.engine.application_factory.html#niftynet.engine.application_factory.ApplicationNetFactory))
for NiftyNet's default network modules.

######  `activation_function`
Sets the type of activation of the network.  Available choices are listed in
`SUPPORTED_OP` in [activation layer](https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/activation.py).
Depending on its implementation, the network might ignore this option .

######  `batch_size`
Sets number of image windows to be processed at each iteration.
When `num_gpus` is greater than 1, `batch_size` is used for each computing device.
That is, the effective inputs at each iteration become `batch_size` x `num_gpus`.

######  `smaller_final_batch_mode`
When total number of window samples are not divisible by batch_size
the class supports different modes for the final batch:
- `drop`: drop the remainder batch
- `pad`: padding the final smaller batch with -1
- `dynamic`: output the remainder directly (
    in this case the batch_size is undetermined at "compile time")

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

See also: [Patch-base analysis guide](./window_sizes.html)

###### `volume_padding_mode`
Set which type of numpy padding to do, see
[https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.pad.html](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.pad.html) for details.

###### `window_sampling`
Type of sampler used to generate image windows from each image volume:
- uniform: fixed size uniformly distributed,
- weighted: fixed size where the likelihood of sampling a voxel is proportional to the cumulative intensity histogram,
- balanced: fixed size where each label has the same probability of being sampled,
- resize: resize image to the window size.

For `weighted` and `balanced`, an input section is required to load sampling priors.
As an [example in the demo folder](https://github.com/NifTK/NiftyNet/blob/v0.3.0/demos/PROMISE12/promise12_balanced_train_config.ini#L61),
`sampler` parameter is set to `label`, indicating that the sampler uses `label`
section as the sampling prior.

###### `queue_length`
Integer specifies window buffer size used when sampling image windows from image volumes.
Image window samplers fill the buffer and networks read the buffer.
Because the network reads [batch_size](#batch-size) windows at each iteration,
this value is set to at least `batch_size * 2.5` to allow for a possible randomised buffer,
i.e. `max(queue_length, round(batch_size * 2.5))`.

###### `keep_prob`
The probability that each element is kept if dropout is supported by the network.
The default value is `0.5`, meaning randomly dropout at the ratio of 0.5.
This is also used as a default value at inference stage.

To achieve a deterministic inference, set `keep_prob=1`;
to draw stochastic samples at inferece, set `keep_prob` to a value in between 0 and 1.

In the case of drawing multiple Monte Carlo samples, the user can run the
inference command mutiple times, with each time a different `save_seg_dir`, for
example:

`python net_segment.py inference ... --save_seg_dir run_2 --keep_prob 0.5`.



##### Volume-normalisation
Intensity based volume normalisation can be configured using a combination of parameters described below:

(1) Setting `normalisation=True` enables the [histogram-based
normalisation](./niftynet.utilities.histogram_standardisation.html).  The
relevant configuration parameters are:
> `histogram_ref_file`, `norm_type`, `cutoff`, `normalise_foreground_only`, `foreground_type`, `multimod_foreground_type`.

These parameters are ignored and histogram-based normalisation is disabled if `normalisation=False`.

(2) Setting `whitening=True` enables the volume level normalisation computed by `(I - mean(I))/std(I)`.
The relevant configuration parameters are:
> `normalise_foreground_only`, `foreground_type`, `multimod_foreground_type`.

These parameters are ignored and whitening is disabled if `whitening=False`.

More specifically:


 Name | Type | Example | Default
 ---- | ---- | ------- | -------
[normalisation](#normalisation) | `boolean` | `normalisation=True` | `False`
[whitening](#whitening) | `boolean` | `whitening=True` | `False`
[histogram_ref_file](#histogram-ref-file) | `string` | `histogram_ref_file=./hist_ref.txt` | `''`
[norm_type](#norm-type) | `string` | `norm_type=percentile` | `percentile`
[cutoff](#cutoff) | `float array (two elements)` | `cutoff=0.1, 0.9` | `0.01, 0.99`
[normalise_foreground_only](#normalise-foreground-only) | `boolean` | `normalise_foreground_only=True` | `False`
[foreground_type](#foreground-type) | `string` | `foreground_type=otsu_plus` | `otsu_plus`
[multimod_foreground_type](#multimod-foreground-type) | `string` | `multimod_foreground_type=and` | `and`

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
> `otsu_plus`, `otsu_minus`, `thresh_plus`, `thresh_minus`, `mean_plus`.

###### `multimod_foreground_type`
Strategies applied to combine foreground masks of multiple modalities, can take one of the following:
* `or` union of the available masks,
* `and` intersection of the available masks,
* `all` masks computed from each modality independently.


### TRAINING

 Name | Type | Example | Default
 ---- | ---- | ------- | -------
[optimiser](#optimiser) | `string` | `optimiser=momentum` | `adam`
[sample_per_volume](#sample-per-volume) | `positive integer` | `sample_per_volume=5` | `1`
[lr](#lr) | `float` | `lr=0.001` | `0.1`
[loss_type](#loss-type) | `string` | `loss_type=CrossEntropy` | `Dice`
[starting_iter](#starting-iter) | `integer` | `starting_iter=0` | `0`
[save_every_n](#save-every-n) | `integer` | `save_every_n=5` | `500`
[tensorboard_every_n](#tensorboard-every-n) | `integer` | `tensorboard_every_n=5` | `20`
[max_iter](#max-iter) | `integer` | `max_iter=1000` | `10000`
[max_checkpoints](#max-checkpoints) | `integer` | `max_checkpoints=5` | `100`
[vars_to_restore](#vars-to-restore) | `string` | `vars_to_restore=^.*(conv_1|conv_2).*$` | `''`
[vars_to_freeze](#vars-to-freeze) | `string` | `vars_to_freeze=^.*(conv_3|conv_4).*$` | value of `vars_to_restore`

###### `optimiser`
Type of optimiser for computing graph gradients.  Current available options are
defined here:
[`SUPPORTED_OPTIMIZERS`](./niftynet.engine.application_factory.html#niftynet.engine.application_factory.OptimiserFactory).

###### `sample_per_volume`
Set number of samples to take from each image volume.

###### `lr`
The learning rate for the optimiser.

###### `loss_type`
Type of loss function.
Please see the relevant loss function layer for choices available:
- [Segmentation](./niftynet.layer.loss_segmentation.html),
- [Regression](./niftynet.layer.loss_regression.html),
- [Autoencoder](./niftynet.layer.loss_autoencoder.html),
- [GAN](./niftynet.layer.loss_gan.html).

The corresponding loss function type names are defined in the
[`ApplicationFactory`](https://github.com/NifTK/NiftyNet/blob/dev/niftynet/engine/application_factory.py)


###### `starting_iter`
The iteration to resume training model.
Setting `starting_iter=0` starts the network from random initialisations.
Setting `starting_iter=-1` starts the network from the latest checkpoint if it exists.

###### `save_every_n`
Frequency of saving the current training model saving.
Setting to a `0` to disable the saving schedule.
(A final model will always be saved when quitting the training loop.)

###### `tensorboard_every_n`
Frequency of evaluating graph elements and write to tensorboard.
Setting to `0` to disable the tensorboard writing schedule.

###### `max_iter`
Maximum number of training iterations.
The value is total number of iterations.
Setting both `starting_iter` and `max_iter` to `0` to
save the random model initialisation.

###### `max_checkpoints`
Maximum number of recent checkpoints to keep.

###### `vars_to_restore`
Regular expression string to match variable names,
values of the matched variables will be initialised for a checkpoint file.

See also: [guide for finetuning pre-trained networks](./transfer_learning.html)

###### `vars_to_freeze`
Regular expression string to match variable names,
values of the matched variables will be updated during training.
Defaulting to the value of `vars_to_restore`.

See also: [guide for finetuning pre-trained networks](./transfer_learning.html)

##### Validation during training
Setting [`validation_every_n`](#validation-every-n) to a positive integer
enables validation loops during training.
When validation is enabled, images list (defined by [input specifications](#input-data-source-section))
will be treated as the whole dataset, and partitioned into subsets of training, validation, and inference
according to [`exclude_fraction_for_validation`](#exclude-fraction-for-validation) and
[`exclude_fraction_for_inference`](#exclude-fraction-for-inference).

A CSV table randomly mapping each file name to one of the stages `{'Training',
'Validation', 'Inference'}` will be generated and written to
[dataset_split_file](#dataset-split-file). This file will be created at the
beginning of training (`starting_iter=0`) and only if the file does not exist.

- If a new random partition is required, please remove the existing [dataset_split_file](#dataset-split-file).

- If no partition is required, please remove any existing [dataset_split_file](#dataset-split-file),
and make sure both [`exclude_fraction_for_validation`](#exclude-fraction-for-validation)
and [`exclude_fraction_for_inference`](#exclude-fraction-for-inference) are `0`.

To exclude particular subjects or adjust the randomly generated partition, the
[`dataset_split_file`](#dataset-split-file) can be edited manually.  Please note
duplicated rows are not removed. For example, if the content of
[`dataset_split_file`](#dataset-split-file) is as follows:
```text
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

During inference, if a [dataset_split_file](#dataset-split-file) is available,
only image files in the `Inference` phase will be used, otherwise inference
will process all image files defined by [input specifications](#input-data-source-section).


 Name | Type | Example | Default
 ---- | ---- | ------- | -------
[validation_every_n](#validation-every-n) | `integer` | `validation_every_n=10` | `-1`
[validation_max_iter](#validation-max-iter) | `integer` | `validation_max_iter=5` | `1`
[exclude_fraction_for_validation](#exclude-fraction-for-validation) | `float` | `exclude_fraction_for_validation=0.2` | `0.0`
[exclude_fraction_for_inference](#exclude-fraction-for-inference) | `float` | `exclude_fraction_for_inference=0.1` | `0.0`

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


 Name | Type | Example | Default
 ---- | ---- | ------- | -------
[rotation_angle](#rotation-angle) | `float array` | `rotation_angle=-10.0,10.0` | `''`
[scaling_percentage](#scaling-percentage) | `float array` | `scaling_percentage=-20.0,20.0` | `''`
[antialiasing](#scaling-percentage) | `boolean` | `antialiasing=True` | `True`
[random_flipping_axes](#random-flipping-axes) | `integer array` | `random_flipping_axes=1,2` | `-1`
[do_elastic_deformation](#do-elastic-deformation) | `boolean` | `do_elastic_deformation=True` | `False`
[num_ctrl_points](#do-elastic-deformation) | `integer` | `num_ctrl_points=1` | `4`
[deformation_sigma](#do-elastic-deformation) | `float` | `deformation_sigma=1` | `15`
[proportion_to_deform](#do-elastic-deformation) | `float` | `proportion_to_deform=0.7` | `0.5`
[bias_field_range](#bias-field-range) | `float array` | `bias_field_range=-10.0,10.0` | `''`
[bf_order](#bias-field-range) | `integer` | `bf_order=1` | `3`


###### `rotation_angle`
Float array, indicates a random rotation operation should be applied to the
volumes (This can be slow depending on the input volume dimensionality).

###### `scaling_percentage`
Float array indicates a random spatial scaling should be applied
(This can be slow depending on the input volume dimensionality).
The option accepts percentages relative to 100 (the original input size).
E.g, `(-50, 50)` indicates transforming
image (size `d`) to image with its size in between `0.5*d` and `1.5d`.

When random scaling is enabled, it is possible to further specify:
- `antialiasing` indicating if antialiasing should be performed
when randomly downsampling the input images.

###### `random_flipping_axes`
The axes which can be flipped to augment the data.
Supply as comma-separated values within single quotes, e.g. '0,1'.
Note that these are 0-indexed, so choose some combination of 0, 1.

###### `do_elastic_deformation`
Boolean value indicates data augmentation using elastic deformations

When `do_elastic_deformation=True`, it is possible to further specify:
- `num_ctrl_points` -- number of control points for the elastic deformation,
- `deformation_sigma` -- the standard deviation for the elastic deformation,
- `proportion_to_deform` -- what fraction of samples to deform elastically.

###### `bias_field_range`
Float array, indicates data augmentation with randomised bias field

When `bias_field_range` is not None, it is possible to further specify:
- `bf_order` -- maximal polynomial order to use for the bias field augmentation.


### INFERENCE

 Name | Type | Example | Default
 ---- | ---- | ------- | -------
[spatial_window_size](#spatial-window-size) | `integer array` | `spatial_window_size=64,64,64` | `''`
[border](#border) | `integer array` | `border=5,5,5` | `0, 0, 0`
[inference_iter](#inference-iter) | `integer` | `inference_iter=1000` | `-1`
[save_seg_dir](#save-seg-dir) | `string` | `save_seg_dir=output/test` | `output`
[output_postfix](#output-postfix) | `string` | `output_postfix=_output` | `_niftynet_out`
[output_interp_order](#output-interp-order) | `non-negative integer` | `output_interp_order=0` | `0`
[dataset_to_infer](#dataset-to-infer) | `string` | `dataset_to_infer=training` | `''`

###### `spatial_window_size`
Array of integers indicating the size of input window.  By default, the window
size at inference time is the same as the [input source specification](#input-data-source-section).
If this parameter is specified, it
overrides the `spatial_window_size` parameter in input source sections.

See also: [Patch-base analysis guide](./window_sizes.html)

###### `border`
Tuple of integers specifying a border size used to crop (along both sides of each
dimension) the network output image window. E.g., `3, 3, 3` will crop a
`64x64x64` window to size `58x58x58`.

See also: [Patch-base analysis guide](./window_sizes.html)

###### `inference_iter`
Integer specifies the trained model to be used for inference.
`-1` or unspecified indicating to use the latest available trained model in `model_dir`.

###### `save_seg_dir`
Prediction directory name. If it's a relative path, it is set to be relative to [`model_dir`](#model-dir).

###### `output_postfix`
Postfix appended to every inference output filenames.

###### `output_interp_order`
Interpolation order of the network outputs.

###### `dataset_to_infer`
String specifies which dataset ('all', 'training', 'validation', 'inference') to compute inference for.
By default 'inference' dataset is used.


### EVALUATION
For evaluation of the output of an application against some available ground
truth, an `EVALUATION` section must be present. Examples of evaluation config
files are available in [the config
folder](https://github.com/NifTK/NiftyNet/tree/dev/config) with the suffix
`_eval.ini`.

The evaluation command is:
```bash
# command to run from git-cloned NiftyNet source code folder
python net_run.py evaluation -c <path_to/config.ini> -a <application>
```
(For example, multimodal segmentation evaluation could be:
`python net_run.py evaluation -a niftynet.applications.segmentation_application.SegmentationApplication -c config/default_multimodal_segmentation_eval.ini`)

In order to run the evaluation, the input sources section must contain the
details on
* The ground truth against which to compare (label in case of the segmentation)
* The corresponding files to evaluate (inferred)

The final evaluation file is saved in the folder indicated as input of the
field `model_dir` in the section `[SYSTEM]` under the form of a csv file with
indication of subject id (label if relevant) and the calculated metrics as
columns.

The evaluation configuration section (`[EVALUATION]`) must contain:
- `save_csv_dir` -- Path where to save the csv file output
- `evaluations` -- List of metrics of evaluation to be calculated presented as a
 string separated by commas (e.g. `dice`, `jaccard`, `n_pos_ref`, `n_pos_seg`).
 Lists of possible evaluations metrics per application are available in
 [regression evaluations](./niftynet.evaluation.regression_evaluations.html),
 [segmentation evaluations](./niftynet.evaluation.segmentation_evaluations.html),
 and [classification evaluations](./niftynet.evaluation.classification_evaluations.html).
- `evaluation_units` -- `foreground`, `label` or `cc`. Describe how the
  evaluation should be performed in the case of segmentation mostly
  (`foreground` means only one label, `label` means metrics per label, `cc`
  means metrics per connected component).  More on this topic can be found at
  [segmentation evaluations](./niftynet.evaluation.segmentation_evaluations.html).
