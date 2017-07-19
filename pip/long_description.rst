NiftyNet
========

NiftyNet is a `TensorFlow`_-based open-source convolutional neural networks (CNN) platform for research in medical image analysis and computer-assisted intervention.
NiftyNet is a consortium of multiple research groups (WEISS -- `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, CMIC -- `Centre for Medical Image Computing`_, HIG -- High-dimensional Imaging Group), where WEISS acts as a consortium lead.
**NiftyNet is not intended for clinical use**.

Features
========

* Easy-to-customise interfaces of network components
* Designed for sharing networks and pretrained models
* Designed to support 2-D, 2.5-D, 3-D, 4-D inputs (2.5-D: volumetric images processed as a stack of 2D slices; 4-D: co-registered multi-modal 3D volumes)
* Efficient discriminative training with multiple-GPU support
* Implemented recent networks (HighRes3DNet, 3D U-net, V-net, DeepMedic)
* Comprehensive evaluation metrics for medical image segmentation

Getting started and contributing
================================

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

Copyright 2017 NiftyNet Contributors.
NiftyNet is released under the Apache License, Version 2.0.
Please see the LICENSE file in the `NiftyNet source code repository`_ for details.

Acknowledgements
================

This project is grateful for the support from the `Wellcome Trust`_, the `Engineering and Physical Sciences Research Council (EPSRC)`_, the `National Institute for Health Research (NIHR)`_, the `Department of Health (DoH)`_, `University College London (UCL)`_, the `Science and Engineering South Consortium (SES)`_, the `STFC Rutherford-Appleton Laboratory`_, and `NVIDIA`_.

.. _`TensorFlow`: https://www.tensorflow.org/
.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/surgical-interventional-sciences
.. _`NiftyNet source code repository`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet
.. _`Centre for Medical Image Computing`: http://cmic.cs.ucl.ac.uk/
.. _`Centre for Medical Image Computing (CMIC)`: http://cmic.cs.ucl.ac.uk/
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome Trust`: https://wellcome.ac.uk/
.. _`Engineering and Physical Sciences Research Council (EPSRC)`: https://www.epsrc.ac.uk/
.. _`National Institute for Health Research (NIHR)`: https://www.nihr.ac.uk/
.. _`Department of Health (DoH)`: https://www.gov.uk/government/organisations/department-of-health
.. _`Science and Engineering South Consortium (SES)`: https://www.ses.ac.uk/
.. _`STFC Rutherford-Appleton Laboratory`: http://www.stfc.ac.uk/about-us/where-we-work/rutherford-appleton-laboratory/
.. _`NVIDIA`: http://www.nvidia.com
