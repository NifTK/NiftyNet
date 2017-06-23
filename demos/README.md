### Usage
##### (a) To run a toy example
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
##### (b) To customise configurations
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
