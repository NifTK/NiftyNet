NiftyNet
========

NiftyNet is an open-source convolutional neural networks (CNN) platform for research in medical image analysis and computer-assisted intervention.
NiftyNet is an initiative of the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_.
**NiftyNet is not intended for clinical use**.

Features
========

* Easy-to-customise interfaces of network components
* Designed for sharing networks and pretrained models
* Designed to support 2-D, 2.5-D, 3-D, 4-D inputs (2.5-D: volumetric images processed as a stack of 2D slices; 4-D: co-registered multi-modal 3D volumes)
* Efficient discriminative training with multiple-GPU support
* Implemented recent networks (HighRes3DNet, 3D U-net, V-net, DeepMedic)
* Comprehensive evaluation metrics for medical image segmentation

Getting started
===============

Please follow the instructions on the `NiftyNet source code repository`_.

Citing NiftyNet
===============

If you use NiftyNet, please cite the following paper:

::

  @InProceedings{niftynet17,
    author = {Li, Wenqi and Wang, Guotai and Fidon, Lucas and Ourselin, Sebastien and Cardoso, M. Jorge and Vercauteren, Tom},
    title = {On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task},
    booktitle = {International Conference on Information Processing in Medical Imaging (IPMI)},
    year = {2017}
  }

Licensing and copyright
=======================

Copyright (c) 2017, University College London.
NiftyNet is released under the Apache License, Version 2.0.
Please see the LICENSE file in the `NiftyNet source code repository`_ for details.

Acknowledgements
================

This project was supported through an Innovative Engineering for Health award by the Wellcome Trust and EPSRC (WT101957, NS/A000027/1), the National Institute for Health Research University College London Hospitals Biomedical Research Centre (NIHR BRC UCLH/UCL High Impact Initiative), UCL EPSRC CDT Scholarship Award (EP/L016478/1), a UCL Overseas Research Scholarship, a UCL Graduate Research Scholarship, and the Health Innovation Challenge Fund by the Department of Health and Wellcome Trust (HICF-T4-275, WT 97914).
The authors would like to acknowledge that the work presented here made use of Emerald, a GPU-accelerated High Performance Computer, made available by the Science & Engineering South Consortium operated in partnership with the STFC Rutherford-Appleton Laboratory.

.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/surgical-interventional-sciences
.. _`NiftyNet source code repository`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet
.. _`Centre for Medical Image Computing`: http://cmic.cs.ucl.ac.uk/
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
