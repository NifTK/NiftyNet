.. NiftyNet documentation master file, created by
   sphinx-quickstart on Wed Aug 30 14:13:50 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome
=======
NiftyNet is a `TensorFlow`_-based open-source convolutional neural networks platform
for research in medical image analysis and image-guided therapy.
NiftyNet's modular structure is designed for sharing networks and pre-trained models.
NiftyNet is a consortium of research groups
(WEISS -- `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_,
CMIC -- `Centre for Medical Image Computing`_,
HIG -- High-dimensional Imaging Group),
where WEISS acts as the consortium lead.

Using NiftyNet's modular structure you can:

* Get started with established pre-trained networks using built-in tools
* Adapt existing networks to your imaging data
* Quickly build new solutions to your own image analysis problems


The code is available via `GitLab`_ and `GitHub`_,
or you can quickly get started with the released versions in the
`Python Package Index`_.
You can also check out the `release notes`_.

.. _`GitLab`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet
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
.. _`NiftyNet model zoo`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNetExampleServer/blob/master/model_zoo.md
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
   model_zoo
   extending_app
   extending_net
   contributing


Resources
=========

`NiftyNet website`_

`Source code on CmicLab`_

`Source code mirror on GitHub`_

`Model zoo repository`_

Mailing list: nifty-net@live.ucl.ac.uk

`Stack Overflow`_ (for general questions)


.. _`NiftyNet website`: http://niftynet.io/
.. _`Source code on CmicLab`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet
.. _`Source code mirror on GitHub`: https://github.com/NifTK/NiftyNet
.. _`Model zoo repository`: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNetExampleServer/blob/master/model_zoo.md
.. _`Stack Overflow`: https://stackoverflow.com/questions/tagged/niftynet


APIs & reference
================

.. toctree::
   :maxdepth: 2

   list_modules


Licensing and copyright
=======================

Copyright 2018 University College London and the NiftyNet Contributors.
NiftyNet is released under the Apache License, Version 2.0.
Please see the `LICENSE file`_ in the `NiftyNet source code repository`_ for details.


Acknowledgements
================

This project is grateful for the support from the `Wellcome Trust`_,
the `Engineering and Physical Sciences Research Council (EPSRC)`_,
the `National Institute for Health Research (NIHR)`_,
the `Department of Health (DoH)`_, `University College London (UCL)`_,
the `Science and Engineering South Consortium (SES)`_,
the `STFC Rutherford-Appleton Laboratory`_, and `NVIDIA`_.

.. _`LICENSE file`: https://github.com/NifTK/NiftyNet/blob/dev/LICENSE
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
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


