# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from io import open

import versioneer
from niftynet.utilities.versioning import get_niftynet_version

niftynet_version = get_niftynet_version()

# Get the summary
description = 'An open-source convolutional neural networks platform' + \
              ' for research in medical image analysis and' + \
              ' image-guided therapy'

# Get the long description
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='NiftyNet',

    version=niftynet_version,
    cmdclass=versioneer.get_cmdclass(),

    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='http://niftynet.io/',

    author='NiftyNet Consortium',
    author_email='niftynet-team@googlegroups.com',

    license='Apache 2.0',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],

    packages=find_packages(
        exclude=[
            'pip',
            'config',
            'data',
            'demos',
            'tests',
        ]
    ),

    install_requires=[
        'six>=1.10',
        'nibabel>=2.1.0',
        'numpy>=1.13.3',
        'scipy>=0.18',
        'configparser',
        'pandas',
        'pillow',
        'blinker',
        'packaging'
    ],

    entry_points={
        'console_scripts': [
            'net_segment=niftynet:main',
            'net_download=niftynet.utilities.download:main',
            'net_run=niftynet:main',
            'net_regress=niftynet:main',
            'net_gan=niftynet:main',
            'net_autoencoder=niftynet:main',
            'net_classify=niftynet:main',
        ],
    },
)
