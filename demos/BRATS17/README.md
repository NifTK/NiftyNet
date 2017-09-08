This demo presents the brain tumor segmentation method described in
```
Wang et al., Automatic Brain Tumor Segmentation using
Cascaded Anisotropic Convolutional Neural Networks,
https://arxiv.org/abs/1709.00382
```

### Overview
The demo focuses on the first stage of the cascaded CNNs, i.e., automated
segmentation of whole tumor using the [WNet](anisotropic_nets/wt_net.py).

This folder (also zipped and downloadable via [dropbox link](http://link)) contains details for replicating the results [1], including:
  * [brats_segmentation.py](brats_segmentation.py) --
    an application built with NiftyNet, defines the main workflow of network
    training and inference.
  * [wt_net.py](anisotropic_nets/wt_net.py) --
    the network definitions.
  * .ini files --
    configuration files define system parameters for running
    segmentation networks, the six files correspond to the combinations of
    [training, inference] and networks in three orientations
    [axial, coronal, sagittal] configurations.
  * [label_mapping_whole_tumor.txt](label_mapping_whole_tumor.txt) --
    mapping file used by NiftyNet, to convert the multi-class segmentations
    into a binary problem.
  * [rename_crop_BRATS17.py](rename_crop_BRATS17.py) --
    utility script used to rename BRATS17 images into
    `TYPEindex_modality.nii.gz` format and crop with a bounding box to remove
    image background (voxels with intensity value zero).

*[1] This implementation ranked the first (in terms of averaged averaged Dice score 0.90499) according
to the online validation leaderboard of [BRATS challenge 2017](https://www.cbica.upenn.edu/BraTS17/lboardValidation.html).*

### Preparing data
This demo requires
[The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)](http://10.1109/TMI.2014.2377694).

To access the data for

 * BRATS 2015, please visit [https://sites.google.com/site/braintumorsegmentation/home/brats2015](https://sites.google.com/site/braintumorsegmentation/home/brats2015).

 * BRATS 2017, please visit [http://www.med.upenn.edu/sbia/brats2017.html](http://www.med.upenn.edu/sbia/brats2017.html).

To be compatiable with the current NiftyNet configuration files and anisotropic
networks, the downloaded datasets must first be preprocessed with [rename_crop_BRATS17.py](rename_crop_BRATS17.py).

### Running segmentation app as a NiftyNet module
Using pip installed NiftyNet:
```
pip install NiftyNet
net_run train -c train_whole_tumor_sagittal.ini --app brats_segmentation.BRATSApp --name anisotropic_nets.wt_net.WTNet
net_run inference -c inference_whole_tumor_sagittal.ini --app brats_segmentation.BRATSApp --name anisotropic_nets.wt_net.WTNet
```
or Using NiftyNet cloned from [GitHub](https://github.com/NifTK/NiftyNet) or [CMICLab](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet):
```
python net_run.py train -c train_whole_tumor_sagittal.ini --app brats_segmentation.BRATSApp --name anisotropic_nets.wt_net.WTNet
python net_run.py inference -c inference_whole_tumor_sagittal.ini --app brats_segmentation.BRATSApp --name anisotropic_nets.wt_net.WTNet
```

