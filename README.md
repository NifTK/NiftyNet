# NiftyNet
NiftyNet is an open-source library for convolutional networks in medical image analysis.

### Features:
* [x] Easy-to-customise interfaces of network components
* [x] Designed for sharing networks and pretrained models with the community
* [x] Efficient discriminative training with multiple-GPU support
* [ ] Comprehensive evaluation metrics for medical image segmentation
* [ ] Implemented recent networks (HighRes3DNet, 3D U-net, V-net, DeepMedic)


### Dependencies:
* Python 2.7
* Tensorflow 1.0
* Nibabel 2.1
* Numpy  1.12
* Scipy 0.19


### Usage
To train a "toynet" specified in `network/toynet.py`:
``` sh
cd NiftyNet/
python run_application.py train --net_name toynet \
--train_image_dir ./example_volumes/T1 --train_label_dir ./example_volumes/Label \
--image_size 42 --label_size 42 --batch_size 1
```
To do segmentation with a trained "toynet":
``` sh
cd NiftyNet/
python run_application.py inference --net_name toynet \
--eval_image_dir ./example_volumes/T1 --save_seg_dir ./seg_output \
--image_size 128 --label_size 128 --batch_size 4
```
*Commandline parameters override the default settings defined in `config/default_config.txt`.*

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

To develop a new architecture:
1. Create a `network/new_net.py` inheriting `network/net_template.py`
1. Implement `inference()` function using the building blocks in `base_layer.py` or creating new building blocks
1. Add `import network.new_net` to the `NetFactory` class in `run_application.py`


### Structure
The basic picture of training procedure (data parallelism) is:
```
<Multi-GPU training>                      (training.py)
                                 |>----------------------+
                                 |>---------------+      |
                                 |^|              |      |
                      (queue.py) |^|     sync   GPU_1  GPU_2   ...
                                 |^|     +----> model  model (network/*.py)
with multiple threads:           |^|     |        |      |
               (preprocess.py)   |^|    CPU       v      v (loss.py)
image&label -->> (sampler.py) -->> |   model <----+------+
(*.nii.gz)   (data_augmentation.py)    update     stochastic gradients
```
