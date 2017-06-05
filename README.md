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
* Python
* Tensorflow
* Nibabel
* Numpy
* Scipy
* configparser
* scikit-image

Please run `pip install -r requirements` to install the dependencies

To intall tensorflow, please follow
https://www.tensorflow.org/install/


### Usage
##### (a) To run a toy example
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

### CMIC Cluster
For UCL CMIC members:
To run NiftyNet on the CS cluster, follow these instructions:

1) If you do not already have a CS cluster account, get one by following [these instructions](http://hpc.cs.ucl.ac.uk/account_application_form/).

2) Log in to the CS cluster by following [these instructions](http://hpc.cs.ucl.ac.uk/how_to_login/):

3) Create a submission script (```mySubmissionScript.sh``` in this example) for the NiftyNet task (```run_application.py train --net_name toynet --image_size 42 --label_size 42 --batch_size 1``` in this example):

```
#$ -P gpu
#$ -l gpu=1
#$ -l gpu_titanxp=1
#$ -l h_rt=23:59:0
#$ -l tmem=11.5G
#$ -S /bin/bash
#!/bin/bash
# The lines above are resource requests. This script has requested 1 Titan X (Pascal) GPU for 24 hours, and 11.5 GB of memory to be started with the BASH Shell. 
# More information about resource requests can be found at http://hpc.cs.ucl.ac.uk/job_submission_sge/

# This line ensures that you only use the 1 GPU requested. 
export CUDA_VISIBLE_DEVICES=$(( `nvidia-smi | grep " / .....MiB"|grep -n " ...MiB / [0-9]....MiB"|cut -d : -f 1|head -n 1` - 1 ))

# If CUDA_VISIBLE_DEVICES is set to -1, there were no available GPUs. This is often due to someone else failing to correctly limit their GPU usage as in the line above.
if (( $CUDA_VISIBLE_DEVICES > -1 ))
then

# These lines backup up you old library path and set a special library path that makes it possible to run tensorflow
export LD_LIBRARY_BACKUP=${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH="/share/apps/libc6_2.17/lib/x86_64-linux-gnu/:/share/apps/libc6_2.17/usr/lib64/:/share/apps/gcc-6.2.0/lib64:/share/apps/gcc-6.2.0/lib:/share/apps/python-3.6.0-shared/lib:/share/apps/cuda-8.0/lib64:/share/apps/cuda-8.0/extras/CUPTI/lib64:${LD_LIBRARY_PATH}" 

# This is the line that runs the NiftyNet command
/share/apps/libc6_2.17/lib/x86_64-linux-gnu/ld-2.17.so $(command -v /share/apps/python-3.6.0-shared/bin/python3) run_application.py train --net_name toynet --image_size 42 --label_size 42 --batch_size 1

# This line restores your path so that you can run normal programs again
export LD_LIBRARY_PATH="${LD_LIBRARY_BACKUP}"

fi
```

4) While logged in to comic100 or comic2, submit the job using qsub

```
qsub mySubmissionScript.sh
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

