# Global NiftyNet settings

The global NiftyNet configuration is read from `~/.niftynet/config.ini`.
When NiftyNet is run, it will attempt to load this file for the global configuration.
* If it does not exist, NiftyNet will create a default one.
* If it exists but cannot be read for some reason (for instance incorrect formatting or wrong entries):
   - NiftyNet will back it up with a timestamp (for instance `~/.niftynet/config-backup-2017-10-16-10-50-58-abc.ini` - `abc` being a random string) and,
   - Create a default one.
* Otherwise NiftyNet will read the global configuration from this file.

Currently the minimal version of this file will look like the following:
```ini
[global]
home = ~/niftynet
```

The `home` key specifies the root folder to be used by NiftyNet for storing and locating user data such as downloaded models, and new networks implemented by the user.
This setting is configurable, and upon successfully loading this file NiftyNet will attempt to create the specified folder, if it does not already exist.


# Configuration file

To run a NiftyNet [application](../niftynet/application) or a customised
application which implements [BaseApplication](../niftynet/application/base_application.py),
a configuration file needs to be provided, for example, 
by creating a `user_configuration.ini` file and using the file via 
`python net_gan.py --conf user_configuration.ini`.

This folder presents a few examples of configuration files for different
applications. All files should have four sections:
- `[SYSTEM]`
- `[NETWORK]`
- `[TRAINING]`
- `[INFERENCE]` 

These describes common options and hyperparameters for all applications.

Additionally, an application specific section is required for each application:
- `[GAN]` for generative adversirial networks
- `[SEGMENTATION]` for segmentation networks
- `[AUTOENCODER]` for autoencoder networks

The [user parameter parser](../niftynet/utilities/user_parameters_parser.py)
tries to match the section names. All other section names will be treated as
[input data source specifications](##Input data source section).


## Input data source section
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
Specifies a set of images (currently supports NIfTI format) 
from `./example_volumes/image_folder`, with filnames contain both `T1` and
`subject`, but not contain `T1c` and `T2`. These images will be read into
memory and transformed into "A, R, S" orientation 
(using [NiBabel](http://nipy.org/nibabel/reference/nibabel.orientations.html)).
The images will also be transformed to have voxel size `(1.0, 1.0, 1.0)`
with an interpolation order of `3`.

This input source can be used alone, as a `T1` MRI input to an application.
It can also be used along with other modalities, a multi-modality example
can be find at [here](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/supports-axbxc-patch/config/default_multimodal_segmentation.ini).

Currently image data in nifty format (extension .nii or .nii.gz) are supported.


The following sections describe key parameters that can be specified in the configuration file.

## [SYSTEM]
* `queue_length` an integer specifies size of image window buffer used when sampling
image windows from image volumes. Image window samplers fill the buffer and
networks read the buffer.
 
## [NETWORK]
### Histogram normalisation
The histogram normalisation is performed using the method described. The following fields can be specified:  
* `normalisation` [True/False] Indicates if an histogram standardisation should be applied to the data
* `whitening` [True/False] Indicates if the loaded image should be whitened I->(I-mean)/std
* `histogram_ref_file`: Name of the file that contains the normalisation parameter if it has been trained before or where to save it
* `norm_type`: type of landmarks used in the histogram for the matching (percentile or quartile)
* `cutoff`: a list of two floats, inferior and superior cutoff of the histograms for the matching
* `foreground_type`: to generate a foreground mask and the normalisation will be applied to foreground only. Choice between:
	* `otsu_plus`
	* `otsu_minus`
	* `thresh_plus`
	* `thresh_minus`  
* `multimod_foreground_type`: strategies applied to combine foreground masks of multiple modalities, can take one of the following:
	* `or` union of the available masks
	* `and` intersection of the available masks
	* `all` a different mask is applied to each modality
	
## [TRAINING]
### Augmentation at training
* `rotation_angle` a tuple of two floats, indicating a random rotation operation should be applied to the volumes
(This can be very slow depending on the input volume dimensionality)
* `scaling_percentage` a tuple of two floats, indicating a random spatial scaling should be applied
(This can be very slow depending on the input volume dimensionality)
* `lr` Learning rate to be applied
* `loss_type`. Loss function to be used
* `reg_type` Regularisor to be used 
* `save_every _n` Frequency of model saving
* `max_iter` Maximum number of training steps
* `volume_padding _size` One side length of the receptive field affected by the network

## [INFERENCE]
* `inference_iter` an integer specifies the trained model to be used for inference.
`-1` or unspecified indicating the latest trained model.
* `spatial_window_size` a tuple of integers indicating the size of input window
at inference time, this overrides the `spatial_window_size` parameter in the input
source sections.
* `border` a tuple of integers specifying a border size used to crop (along both sides of each
dimension) the network output image window. E.g., `(3, 3, 3)` will crop a
`(64, 64, 64)` window to size `(58, 58, 58)`.










