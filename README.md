# NiftyNet
NiftyNet is an open-source library for 3D convolutional networks in medical image analysis.

NiftyNet was developed by the [Centre for Medical Image Computing][cmic] at
[University College London (UCL)][ucl].

### Features
* Easy-to-customise interfaces of network components
* Designed for sharing networks and pretrained models with the community
* Efficient discriminative training with multiple-GPU support
* Comprehensive evaluation metrics for medical image segmentation
* Implemented recent networks (HighRes3DNet, 3D U-net, V-net, DeepMedic)


### Dependencies
* Python 2.7
* Tensorflow 1.0
* Nibabel 2.1
* Numpy 1.12
* Scipy 0.19
* medpy 0.2.2


### Usage
To train a "toynet" specified in `network/toynet.py`:
``` sh
cd NiftyNet/
python run_application.py train --net_name toynet \
    --train_data_dir ./example_volumes/monomodal_parcellation \
    --image_size 42 --label_size 42 --batch_size 1
```
After the training process, to do segmentation with a trained "toynet":
``` sh
cd NiftyNet/
python run_application.py inference --net_name toynet \
    --eval_data_dir ./example_volumes/monomodal_parcellation \
    --save_seg_dir ./seg_output \
    --image_size 64 --label_size 64 --batch_size 4
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

### Files conventions
Files used for training (resp. inference) are supposed to be located in the `train_data_dir`
(resp. `eval_data_dir`) directory and to respect the name convention `patient_modality.extension`.
Please note the delimiter `_`.

Only nifty files (extension .nii or .nii.gz) are supported.

### Structure
The basic picture of training procedure (data parallelism) is:
```
<Multi-GPU training>
                                         (nn/training.py)
                                   |>----------------------+
                                   |>---------------+      |
                                   |^|              |      |
               (nn/input_queue.py) |^|     sync   GPU_1  GPU_2   ...
                                   |^|     +----> model  model (network/*.py)
with multiple threads:             |^|     |        |      |
              (nn/preprocess.py)   |^|    CPU       v      v (nn/loss.py)
image&label ->> (nn/sampler.py)  ->> |   model <----+------+
(*.nii.gz)  (nn/data_augmentation.py)|   update    stochastic gradients
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
