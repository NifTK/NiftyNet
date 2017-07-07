from setuptools import setup, find_packages


desc='NiftyNet is an open-source library for convolutional' \
     ' networks in medical image analysis.'
more='NiftyNet was developed by the Centre for Medical Image' \
     ' Computing at University College London (UCL).' \
     '\n' \
     'Features:' \
     '\n * Easy-to-customise interfaces of network components' \
     '\n * Designed for sharing networks and pretrained models' \
     '\n * Designed to support 2-D, 2.5-D, 3-D, 4-D inputs (' \
     '\n   (2.5-D: volumetric images processed as a stack of 2D' \
     ' slices; 4-D: co-registered multi-modal 3D volumes)' \
     '\n * Efficient discriminative training with multiple-GPU' \
     ' support' \
     '\n * Implemented recent networks (HighRes3DNet, 3D U-net,' \
     ' V-net, DeepMedic)' \
     '\n * Comprehensive evaluation metrics for medical image' \
     ' segmentation'


setup(
    name='NiftyNet',

    version='0.1',  # TODO

    description=desc,
    long_description=more,  # TODO

    url='',  # TODO

    author='',  # TODO
    author_email='',  # TODO

    license='',  # TODO

    classifiers=[
        '',

        '',
    ],  # TODO

    keywords='',  # TODO

    packages=find_packages(exclude=['', '', '']),  # TODO

    install_requires=['', '', ''],  # TODO

    entry_points={
        'console_scripts': [
            '',  # TODO
        ],
    },
)

