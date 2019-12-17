Installation
============

1. Installing the appropriate `TensorFlow`_ package:

    - ``pip install "tensorflow==1.15.*"``

2. Installing NiftyNet package

    Option 1. released version from `PyPI`_:

    .. code-block:: bash

        pip install niftynet

    Option 2. latest dev version from `source code repository`_:

    .. code-block:: bash

        git clone https://github.com/NifTK/NiftyNet.git

        # installing dependencies from the list of requirements
        cd NiftyNet/
        pip install -r requirements.txt

    Alternatively, you can `download the code`_ as a .zip file and extract it.

3. (Optional) Accessing MetaImage format (``.mha/.mhd`` files) requires SimpleITK:

    .. code-block:: bash

      pip install SimpleITK


.. _`TensorFlow`: https://www.tensorflow.org/
.. _`PyPI`: https://pypi.org/project/NiftyNet/
.. _`source code repository`: https://github.com/NifTK/NiftyNet
.. _`download the code`: https://github.com/NifTK/NiftyNet/archive/dev.zip
