# -*- coding: utf-8 -*-
import re

from packaging import version
from setuptools import setup, find_packages

import versioneer
from niftynet.utilities.versioning import get_niftynet_version

niftynet_version = get_niftynet_version()

# Regex for checking PEP 440 conformity
# https://www.python.org/dev/peps/pep-0440/#id79
pep440_regex = re.compile(
    r"^\s*" + version.VERSION_PATTERN + r"\s*$",
    re.VERBOSE | re.IGNORECASE,
)

# Check PEP 440 conformity
if niftynet_version is not None and \
        pep440_regex.match(niftynet_version) is None:
    raise ValueError('The version string {} does not conform to'
                     ' PEP 440'.format(niftynet_version))

# Get the summary
description = 'An open-source convolutional neural networks platform' + \
              ' for research in medical image analysis and' + \
              ' image-guided therapy'

# Get the long description
with open('pip/long_description.rst') as f:
    long_description = f.read()

setup(
    name='NiftyNet',

    version=niftynet_version,
    cmdclass=versioneer.get_cmdclass(),

    description=description,
    long_description=long_description,

    url='http://niftynet.io/',

    author='NiftyNet Consortium',
    author_email='nifty-net@live.ucl.ac.uk',

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
        'numpy>=1.12',
        'scipy>=0.18',
        'configparser',
        'pandas',
        'pillow',
        'blinker'
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
