### Dependencies
* six
* Python
* Tensorflow
* Nibabel
* Numpy
* Scipy
* configparser
* scikit-image

### Usage
##### To install dependencies

Run `pip install -r requirements-gpu.txt` to install all dependencies
with GPU support, 

Run `pip install -r requirements-cpu.txt` for a CPU support
only version.

For more information on installing Tensorflow, please follow
https://www.tensorflow.org/install/

##### (a) To run the demos:
Please see the `README.md` in each folder of this [directory](./demos) for more details.

##### (b) The "run_application" command:
To train a "toynet" specified in `network/toynet.py`:
``` sh
cd NiftyNet/
python run_application.py train --net_name toynet \
    --image_size 42 --label_size 42 --batch_size 1
```
(GPU computing is enabled by default; to train with CPU only please use `--num_gpus 0`)

After the training process, to do segmentation with a trained "toynet":
``` sh
cd NiftyNet/
python run_application.py inference --net_name toynet \
    --save_seg_dir ./seg_output \
    --image_size 80 --label_size 80 --batch_size 8
```

Image data in nifty format (extension .nii or .nii.gz) are supported.

##### (c) To customise configurations
Commandline parameters override the default settings defined in `config/default_config.txt`.

Alternatively, to run with a customised config file:

``` sh
cd NiftyNet/
# training
python run_application.py train -c /path/to/customised_config
# inference
python run_application.py inference -c /path/to/customised_config
```
where `/path/to/customised_config` implements all parameters listed by running:
```sh
python run_application.py -h
```
