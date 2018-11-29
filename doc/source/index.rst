.. NiftyNet documentation master file, created by
   sphinx-quickstart on Wed Aug 30 14:13:50 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome
=======
NiftyNet is a `TensorFlow`_-based open-source convolutional neural networks platform
for research in medical image analysis and image-guided therapy.
NiftyNet's modular structure is designed for sharing networks and pre-trained models.
NiftyNet is a consortium of research organisations
(BMEIS -- `School of Biomedical Engineering and Imaging Sciences, King's College London`_;
WEISS -- `Wellcome EPSRC Centre for Interventional and Surgical Sciences, UCL`_;
CMIC -- `Centre for Medical Image Computing, UCL`_;
HIG -- High-dimensional Imaging Group, UCL),
where BMEIS acts as the consortium lead.

Using NiftyNet's modular structure you can:

* Get started with established pre-trained networks using built-in tools
* Adapt existing networks to your imaging data
* Quickly build new solutions to your own image analysis problems


The code is available via `GitHub`_,
or you can quickly get started with the released versions in the
`Python Package Index`_.
You can also check out the `release notes`_.

.. _`GitHub`: https://github.com/NifTK/NiftyNet
.. _`Python Package Index`: https://pypi.org/project/NiftyNet/
.. _`release notes`: https://github.com/NifTK/NiftyNet/blob/dev/CHANGELOG.md


Quickstart
==========
This section shows you how to run a segmentation application with ``net_segment``
command, using a model with trained weights and image data downloaded
from `NiftyNet model zoo`_ with ``net_download``.

With NiftyNet `installed from PyPI`_:

.. code-block:: bash

    net_download dense_vnet_abdominal_ct_model_zoo
    net_segment inference -c ~/niftynet/extensions/dense_vnet_abdominal_ct/config.ini

With NiftyNet `source code`_ cloned at ``./NiftyNet/``:

.. code-block:: bash

    # go to the source code directory
    cd NiftyNet/
    python net_download.py dense_vnet_abdominal_ct_model_zoo
    python net_segment.py inference -c ~/niftynet/extensions/dense_vnet_abdominal_ct/config.ini

The segmentation output of this example application should be located at

.. code-block:: bash

    ~/niftynet/models/dense_vnet_abdominal_ct/segmentation_output/

.. topic:: Applications and models

  More applications and models are available at `NiftyNet model zoo`_ and the
  `network`_ directory.

.. topic:: Configuration specifications

  For detailed specifications of NiftyNet commands and configurations,
  check out our `Configuration docs`_.

.. topic:: Extending NiftyNet applications

  To learn more about developing NiftyNet applications, see the `Extending
  application`_ and `Developing new networks`_ section.

.. topic:: Contributing to NiftyNet

  Contributors are always welcomed!  For more information please visit the
  `Contributor guide`_ section.

All how-to guides are listed in `the following section <#guides>`_.



.. _`installed from PyPI`: installation.html
.. _`source code`: installation.html
.. _`NiftyNet model zoo`: https://github.com/NifTK/NiftyNetModelZoo/blob/master/README.md
.. _`network`: niftynet.network.html
.. _`Configuration docs`: config_spec.html
.. _`Extending application`: extending_app.html
.. _`Developing new networks`: extending_net.html
.. _`Contributor guide`: contributing.html


Guides
======
.. toctree::
   :maxdepth: 1

   introductory
   installation
   config_spec
   filename_matching
   window_sizes
   model_zoo
   extending_app
   extending_event_handler
   extending_net
   transfer_learning
   contributing


Resources
=========

`NiftyNet website`_

`Source code on GitHub`_

`Model zoo repository`_

Mailing list: niftynet@googlegroups.com

`Stack Overflow`_ (for general questions)


.. _`NiftyNet website`: http://niftynet.io/
.. _`Source code on GitHub`: https://github.com/NifTK/NiftyNet
.. _`Model zoo repository`: https://github.com/NifTK/NiftyNetModelZoo/blob/master/README.md
.. _`Stack Overflow`: https://stackoverflow.com/questions/tagged/niftynet


APIs & reference
================

.. toctree::
   :maxdepth: 2

   list_modules


Licensing and copyright
=======================

NiftyNet is released under `the Apache License, Version 2.0`_.


Copyright 2018 the NiftyNet Consortium.



Acknowledgements
================

This project is grateful for the support from the `Wellcome Trust`_,
the `Engineering and Physical Sciences Research Council (EPSRC)`_,
the `National Institute for Health Research (NIHR)`_,
the `Department of Health (DoH)`_,
`King's College London (KCL)`_,
`University College London (UCL)`_,
the `Science and Engineering South Consortium (SES)`_,
the `STFC Rutherford-Appleton Laboratory`_, and `NVIDIA`_.

.. _`the Apache License, Version 2.0`: https://github.com/NifTK/NiftyNet/blob/dev/LICENSE
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`School of Biomedical Engineering and Imaging Sciences, King's College London`: https://www.kcl.ac.uk/lsm/research/divisions/imaging/index.aspx
.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences, UCL`: http://www.ucl.ac.uk/interventional-surgical-sciences
.. _`Centre for Medical Image Computing, UCL`: http://www.ucl.ac.uk/medical-image-computing
.. _`Centre for Medical Image Computing (CMIC)`: http://www.ucl.ac.uk/medical-image-computing
.. _`King's College London (KCL)`: https://www.kcl.ac.uk/
.. _`University College London (UCL)`: https://www.ucl.ac.uk/
.. _`Wellcome Trust`: https://wellcome.ac.uk/
.. _`Engineering and Physical Sciences Research Council (EPSRC)`: https://epsrc.ukri.org/
.. _`National Institute for Health Research (NIHR)`: https://www.nihr.ac.uk/
.. _`Department of Health (DoH)`: https://www.gov.uk/government/organisations/department-of-health-and-social-care
.. _`Science and Engineering South Consortium (SES)`: https://www.ses.ac.uk/
.. _`STFC Rutherford-Appleton Laboratory`: https://stfc.ukri.org/about-us/where-we-work/rutherford-appleton-laboratory/
.. _`NVIDIA`: http://www.nvidia.com
.. _`NiftyNet source code repository`: https://github.com/NifTK/NiftyNet/blob/dev/LICENSE


If you use NiftyNet in your work, please cite `Gibson and Li, et al. 2017`_:

..

  E. Gibson*, W. Li*, C. Sudre, L. Fidon, D. I. Shakir, G. Wang,
  Z. Eaton-Rosen, R. Gray, T. Doel, Y. Hu, T. Whyntie, P. Nachev, M. Modat,
  D. C. Barratt, S. Ourselin, M. J. Cardoso^ and T. Vercauteren^ 2017.
  `NiftyNet: a deep-learning platform for medical imaging.`_
  Computer Methods and Programs in Biomedicine (2017).

BibTeX entry:

.. code-block:: bibtex

  @InProceedings{niftynet18,
    author = "Eli Gibson and Wenqi Li and Carole Sudre and Lucas Fidon and
              Dzhoshkun I. Shakir and Guotai Wang and Zach Eaton-Rosen and
              Robert Gray and Tom Doel and Yipeng Hu and Tom Whyntie and
              Parashkev Nachev and Marc Modat and Dean C. Barratt and
              Sébastien Ourselin and M. Jorge Cardoso and Tom Vercauteren",
    title = "NiftyNet: a deep-learning platform for medical imaging",
    journal = "Computer Methods and Programs in Biomedicine",
    year = "2018",
    issn = "0169-2607",
    doi = "https://doi.org/10.1016/j.cmpb.2018.01.025",
    url = "https://www.sciencedirect.com/science/article/pii/S0169260717311823",
  }

The NiftyNet platform originated in software developed for `Li, et al. 2017`_:

  Li W., Wang G., Fidon L., Ourselin S., Cardoso M.J., Vercauteren T. (2017)
  `On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task.`_
  In: Niethammer M. et al. (eds) Information Processing in Medical Imaging. IPMI 2017.
  Lecture Notes in Computer Science, vol 10265. Springer, Cham. `DOI: 10.1007/978-3-319-59050-9_28`_

.. _`NiftyNet: a deep-learning platform for medical imaging.`: https://doi.org/10.1016/j.cmpb.2018.01.025
.. _`Gibson and Li, et al. 2017`: https://doi.org/10.1016/j.cmpb.2018.01.025
.. _`Li, et al. 2017`: https://doi.org/10.1007/978-3-319-59050-9_28
.. _`On the Compactness, Efficiency, and Representation of 3D Convolutional Networks: Brain Parcellation as a Pretext Task.`: https://doi.org/10.1007/978-3-319-59050-9_28
.. _`DOI: 10.1007/978-3-319-59050-9_28`: https://doi.org/10.1007/978-3-319-59050-9_28


