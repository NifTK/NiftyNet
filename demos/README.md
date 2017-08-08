### Dependencies
* six
* Python
* Tensorflow
* Nibabel
* Numpy
* Scipy
* configparser

### Usage
##### To install dependencies

Run `pip install -r requirements-gpu.txt` to install all dependencies
with GPU support, 

Run `pip install -r requirements-cpu.txt` for a CPU support
only version.

For more information on installing Tensorflow, please follow
https://www.tensorflow.org/install/

##### (a) Running the demos:
Please see the `README.md` in each folder of this [directory](./demos) for more details.

##### (b) Running a NiftyNet "toynet" example:
To train a "toynet" specified in `network/toynet.py`:
``` sh
cd NiftyNet/
wget -N https://www.dropbox.com/s/y7mdh4m9ptkibax/example_volumes.tar.gz
tar -xzvf example_volumes.tar.gz
net_segment train --net_name toynet \
    --image_size 42 --label_size 42 --batch_size 1
```
(GPU computing is enabled by default; to train with CPU only please use `--num_gpus 0`)

After the training process, to do segmentation with a trained "toynet":
``` sh
cd NiftyNet/
net_segment inference --net_name toynet \
    --save_seg_dir ./seg_output \
    --image_size 80 --label_size 80 --batch_size 8
```

Image data in nifty format (extension .nii or .nii.gz) are supported.

##### (c) To customise configurations
Commandline parameters override the default settings defined in `config/default_config.ini`.

Alternatively, to run with a customised config file:

``` sh
cd NiftyNet/
# training
net_segment train -c /path/to/customised_config
# inference
net_segment inference -c /path/to/customised_config
```
where `/path/to/customised_config` implements all parameters listed by running:
```sh
net_segment -h
```
