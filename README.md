# NiftyNet
<img src="https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/raw/master/niftynet-logo.png" width="263" height="155">

[![build status](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/badges/master/build.svg)](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/commits/master)
[![coverage report](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/badges/master/coverage.svg)](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/commits/master)

NiftyNet is an open-source library for convolutional networks in medical image analysis.

NiftyNet was developed by the [Centre for Medical Image Computing][cmic] at
[University College London (UCL)][ucl].

### Features
* Easy-to-customise interfaces of network components
* Designed for sharing networks and pretrained models
* Designed to support 2-D, 2.5-D, 3-D, 4-D inputs*
* Efficient discriminative training with multiple-GPU support
* Implemented recent networks (HighRes3DNet, 3D U-net, V-net, DeepMedic)
* Comprehensive evaluation metrics for medical image segmentation

 <sup>*2.5-D: volumetric images processed as a stack of 2D slices; 
4-D: co-registered multi-modal 3D volumes</sup>
### Dependencies
* six
* Python 2.7
* Tensorflow 1.0
* Nibabel 2.1
* Numpy 1.12
* Scipy 0.19
* configparser
* scikit-image


### Usage
##### (a) To Run a toy example
To train a "toynet" specified in `network/toynet.py`:
``` sh
cd NiftyNet/
# download demo data (~62MB)
wget https://www.dropbox.com/s/y7mdh4m9ptkibax/example_volumes.tar.gz
tar -xzvf example_volumes.tar.gz
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

##### (c) To develop a new network architecture
1. Create a `network/new_net.py` inheriting `TrainableLayer` from `layer.base_layer`
1. Implement `layer_op()` function using the building blocks in `layer/` or creating new layers
1. Import `network.new_net` to the `NetFactory` class in `run_application.py`
1. Train the network with `python run_application.py train -c /path/to/customised_config`


Image data in nifty format (extension .nii or .nii.gz) are supported.

### Structure
The basic picture of a training procedure (data parallelism) is:
```
<Multi-GPU training>
                                     (engine/training.py)
                                            |>----------------------+
                                            |>---------------+      |
                                            |^|              |      |
                   (engine/input_buffer.py) |^|     sync   GPU_1  GPU_2   ...
                                            |^|     +----> model  model (network/*.py)
with multiple threads:                      |^|     |        |      |
            (layer/input_normalisation.py)  |^|    CPU       v      v (layer/*.py)
image&label ->> (engine/*_sampler.py)   >>>   |   model <----+------+
(*.nii.gz)        (layer/rand_*.py)     >>>   |  update    stochastic gradients
```

### Citation
If you use this software, please cite:
```
@InProceedings{niftynet17,
  author = {Li, Wenqi and Wang, Guotai and Fidon, Lucas and Ourselin, Sebastien and Cardoso, M. Jorge and Vercauteren, Tom},
  title = {On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task},
  booktitle = {International Conference on Information Processing in Medical Imaging (IPMI)},
  year = {2017}
}
```


### Acknowledgements
This project was supported through an Innovative Engineering for Health award by
the Wellcome Trust and EPSRC (WT101957, NS/A000027/1), the National Institute
for Health Research University College London Hospitals Biomedical Research
Centre (NIHR BRC UCLH/UCL High Impact Initiative), UCL EPSRC CDT Scholarship
Award (EP/L016478/1), a UCL Overseas Research Scholarship, a UCL Graduate
Research Scholarship, and the Health Innovation Challenge Fund by the
Department of Health and Wellcome Trust (HICF-T4-275, WT 97914). The authors
would like to acknowledge that the work presented here made use of Emerald, a
GPU-accelerated High Performance Computer, made available by the Science &
Engineering South Consortium operated in partnership with the STFC
Rutherford-Appleton Laboratory.

[cmic]: http://cmic.cs.ucl.ac.uk
[ucl]: http://www.ucl.ac.uk

