This demo presents how to use NiftyNet for whole tumor segmentation,
which is the first stage of the cascaded CNNs described in the following [paper][wang17_paper].
[wang17_paper]: https://arxiv.org/abs/1709.00382

```
Wang et al., Automatic Brain Tumor Segmentation using Cascaded Anisotropic Convolutional Neural Networks, MICCAI BRATS 2017.
```

For a full implementation of the method described in this paper with three stages of the cascaded CNNs, 
please see: https://github.com/taigw/brats17

![A slice from BRATS17](./example_outputs/original.png)
![Ground truth of whole Tumor](./example_outputs/label.png)
![Segmentation probability map using this demo](./example_outputs/ave_prob.png)

From left to right: A slice from BRATS17, ground truth of whole tumor,
and segmentation probability map using this demo [1].

*[1] This method ranked the first (in terms of averaged Dice score 0.90499) according
to the online validation leaderboard of [BRATS challenge 2017](https://www.cbica.upenn.edu/BraTS17/lboardValidation.html).*

_Please see also a trained model in [NiftyNet model zoo](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNetExampleServer/blob/master/anisotropic_nets_brats_challenge_model_zoo.md)._

### Overview
This folder (also zipped and downloadable from
[here](https://www.dropbox.com/s/macplyp53v0tm1j/BRATS17.tar.gz)) contains
details for replicating the results, including:
  * [`brats_segmentation.py`](./brats_segmentation.py) --
    an application built with NiftyNet, defines the main workflow of network
    training and inference.
  * [`wt_net.py`](./anisotropic_nets/wt_net.py) --
    the network definitions.
  * `.ini` files --
    configuration files define system parameters for running
    segmentation networks, the six files correspond to the combinations of
    [training, inference] and networks in three orientations
    [axial, coronal, sagittal] configurations.
  * [`label_mapping_whole_tumor.txt`](./label_mapping_whole_tumor.txt) --
    mapping file used by NiftyNet, to convert the multi-class segmentations
    into a binary problem.
  * [`rename_crop_BRATS.py`](./rename_crop_BRATS.py) --
    utility script used to rename BRATS17 images into
    `TYPEindex_modality.nii.gz` format and crop with a bounding box to remove
    image background (voxels with intensity value zero).


### Preparing data
This demo requires
[The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)](http://10.1109/TMI.2014.2377694).

To access the data for

 * BRATS 2015, please visit [https://sites.google.com/site/braintumorsegmentation/home/brats2015](https://sites.google.com/site/braintumorsegmentation/home/brats2015).

 * BRATS 2017, please visit [http://www.med.upenn.edu/sbia/brats2017.html](http://www.med.upenn.edu/sbia/brats2017.html).

To be compatible with the current NiftyNet configuration files and anisotropic
networks, the downloaded datasets must first be preprocessed with [rename_crop_BRATS.py](./rename_crop_BRATS.py).

### Running segmentation app as a NiftyNet module
Using pip installed NiftyNet:
```bash
pip install NiftyNet
# train WTNet in the sagittal view using BRATSApp
net_run train -c train_whole_tumor_sagittal.ini --app brats_segmentation.BRATSApp --name anisotropic_nets.wt_net.WTNet
# WTNet inference in the sagittal view using BRATSApp
net_run inference -c inference_whole_tumor_sagittal.ini --app brats_segmentation.BRATSApp --name anisotropic_nets.wt_net.WTNet
```
or using NiftyNet cloned from [GitHub](https://github.com/NifTK/NiftyNet) or [CMICLab](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet):
```bash
cd NiftyNet/
# train WTNet in the sagittal view using BRATSApp
python net_run.py train -c train_whole_tumor_sagittal.ini --app brats_segmentation.BRATSApp --name anisotropic_nets.wt_net.WTNet
# WTNet inference in the sagittal view using BRATSApp
python net_run.py inference -c inference_whole_tumor_sagittal.ini --app brats_segmentation.BRATSApp --name anisotropic_nets.wt_net.WTNet
```

##### Note
The above commands require proper configuration of a few file paths:

 * Make sure `brats_segmentation.py` and `anisotropic_nets`
on the `$PYTHONPATH` of the system environment, so that NiftyNet can import the
modules correctly. This can be done by downloading these files, and adding
them to `$PATHONPATH`, for example, given this folder is downloaded
to `/home/BRATS17`:
```bash
export PYTHONPATH=/home/BRATS17:$PYTHONPATH
```

 * In the configuration files (files with extension name `.ini`), `path_to_search`
 needs to be set to the downloaded and preprocessed BRATS dataset;
 `model_dir` and `save_seg_dir` needs to be set to a writable directory; `histogram_ref_file`
 should be pointing at the location of [`label_mapping_whole_tumor.txt`](./label_mapping_whole_tumor.txt).
