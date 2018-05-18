# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from packaging import version
import re
import os

from niftynet.utilities.versioning import get_niftynet_git_version

version_buf, version_git, command_git = get_niftynet_git_version()

# Create a niftynet/info.py module that will keep the
# version descriptor returned by Git
info_module = open(os.path.join('niftynet', 'info.py'), 'w')
info_module.write('# -*- coding: utf-8 -*-\n')
info_module.write('"""NiftyNet version tracker.\n')
info_module.write('\n')
info_module.write('This module only holds the NiftyNet version,')
info_module.write(' generated using the \n')
info_module.write('``{}`` command.\n'.format(' '.join(command_git)))
info_module.write('\n')
info_module.write('"""\n')
info_module.write('\n')
info_module.write('\n')
info_module.write('VERSION_DESCRIPTOR = "{}"\n'.format(version_buf))
info_module.close()

# Regex for checking PEP 440 conformity
# https://www.python.org/dev/peps/pep-0440/#id79
pep440_regex = re.compile(
    r"^\s*" + version.VERSION_PATTERN + r"\s*$",
    re.VERBOSE | re.IGNORECASE,
)

# Check PEP 440 conformity
if pep440_regex.match(version_git) is None:
    raise ValueError('The version tag {} constructed from {} output'
                     ' (generated using the "{}" command) does not'
                     ' conform to PEP 440'.format(
                         version_git, version_buf, ' '.join(command_git)))

# Get the summary
description = 'An open-source convolutional neural networks platform' +\
              ' for research in medical image analysis and' +\
              ' image-guided therapy'

# Get the long description
with open('pip/long_description.rst') as f:
    long_description = f.read()


setup(
    name='NiftyNet',

    version=version_git,

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

