NiftyNet
========

NiftyNet is a `TensorFlow`_-based [#]_ open-source convolutional neural networks (CNN) platform for research in medical image analysis and image-guided therapy.
NiftyNet's modular structure is designed for sharing networks and pre-trained models.
Using this modular structure you can:

* Get started with established pre-trained networks using built-in tools
* Adapt existing networks to your imaging data
* Quickly build new solutions to your own image analysis problems

NiftyNet is a consortium of research groups (WEISS -- `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, CMIC -- `Centre for Medical Image Computing`_, HIG -- High-dimensional Imaging Group), where WEISS acts as the consortium lead.

.. image:: https://badge.fury.io/py/NiftyNet.svg
  :target: https://badge.fury.io/py/NiftyNet

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
  :target: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/LICENSE

.. [#] Please install the appropriate `TensorFlow`_ PyPI package (``tensorflow`` or ``tensorflow-gpu``) **before** executing ``pip install niftynet`` -- see the instructions on the `NiftyNet source code repository`_ for details.


Features
--------

NiftyNet currently supports medical image segmentation and generative adversarial networks.
**NiftyNet is not intended for clinical use**.
Other features of NiftyNet include:

* Easy-to-customise interfaces of network components
* Sharing networks and pretrained models
* Support for 2-D, 2.5-D, 3-D, 4-D inputs [#]_
* Efficient discriminative training with multiple-GPU support
* Implementation of recent networks (HighRes3DNet, 3D U-net, V-net, DeepMedic)
* Comprehensive evaluation metrics for medical image segmentation

NiftyNet release notes are available in the `changelog`_.

.. _`changelog`: https://github.com/NifTK/NiftyNet/blob/dev/CHANGELOG.md

.. [#] 2.5-D: volumetric images processed as a stack of 2D slices; 4-D: co-registered multi-modal 3D volumes


Getting started
---------------

Installation
^^^^^^^^^^^^

Please follow the `installation instructions`_.

.. _`installation instructions`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet#installation

Examples
^^^^^^^^

Please see the `NiftyNet demos`_.

.. _`NiftyNet demos`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/tree/dev/demos

Network (re-)implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please see the list of `network (re-)implementations in NiftyNet`_.

.. _`network (re-)implementations in NiftyNet`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/tree/dev/niftynet/network

API documentation
^^^^^^^^^^^^^^^^^

The API reference is available on `Read the Docs`_.

.. _`Read the Docs`: http://niftynet.rtfd.io/

Contributing
^^^^^^^^^^^^

Please see the `contribution guidelines`_ on the `NiftyNet source code repository`_.

.. _`contribution guidelines`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/CONTRIBUTING.md

Useful links
^^^^^^^^^^^^

`NiftyNet website`_

`NiftyNet source code on CmicLab`_

`NiftyNet source code mirror on GitHub`_

NiftyNet mailing list: nifty-net@live.ucl.ac.uk


.. _`NiftyNet website`: http://niftynet.io/
.. _`NiftyNet source code on CmicLab`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet
.. _`NiftyNet source code mirror on GitHub`: https://github.com/NifTK/NiftyNet


Citing NiftyNet
---------------

If you use NiftyNet in your work, please cite `Li et. al. 2017`_:

  Li W., Wang G., Fidon L., Ourselin S., Cardoso M.J., Vercauteren T. (2017)
  `On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task.`_
  In: Niethammer M. et al. (eds) Information Processing in Medical Imaging. IPMI 2017.
  Lecture Notes in Computer Science, vol 10265. Springer, Cham. DOI: `10.1007/978-3-319-59050-9_28`_

BibTeX entry:

.. code-block:: bibtex

  @InProceedings{niftynet17,
    author = {Li, Wenqi and Wang, Guotai and Fidon, Lucas and Ourselin, Sebastien and Cardoso, M. Jorge and Vercauteren, Tom},
    title = {On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task},
    booktitle = {International Conference on Information Processing in Medical Imaging (IPMI)},
    year = {2017}
  }

.. _`Li et. al. 2017`: http://doi.org/10.1007/978-3-319-59050-9_28
.. _`On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task.`: http://doi.org/10.1007/978-3-319-59050-9_28
.. _`10.1007/978-3-319-59050-9_28`: http://doi.org/10.1007/978-3-319-59050-9_28


Licensing and copyright
-----------------------

Copyright 2018 University College London and the NiftyNet Contributors.
NiftyNet is released under the Apache License, Version 2.0.
Please see the LICENSE file in the `NiftyNet source code repository`_ for details.


Acknowledgements
----------------

This project is grateful for the support from the `Wellcome Trust`_, the `Engineering and Physical Sciences Research Council (EPSRC)`_, the `National Institute for Health Research (NIHR)`_, the `Department of Health (DoH)`_, `University College London (UCL)`_, the `Science and Engineering South Consortium (SES)`_, the `STFC Rutherford-Appleton Laboratory`_, and `NVIDIA`_.

.. _`TensorFlow`: https://www.tensorflow.org/
.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
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
