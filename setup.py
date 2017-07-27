from setuptools import setup, find_packages
from subprocess import check_output


# Describe the version relative to last tag
version_buf = check_output(['git', 'describe', '--match', 'v[0-9]*']).rstrip()
# exclude the 'v' for PEP440 compatibility, see
# https://www.python.org/dev/peps/pep-0440/#public-version-identifiers
version_buf = version_buf[1:]
tokens = version_buf.split('-')
if len(tokens) > 1:  # not a tagged commit
    # Format a developmental release identifier according to PEP440, see:
    # https://www.python.org/dev/peps/pep-0440/#developmental-releases
    version_git = '{}.dev{}'.format(tokens[0], tokens[1])
elif len(tokens) == 1:  # tagged commit
    # Format a public version identifier according to PEP440, see:
    # https://www.python.org/dev/peps/pep-0440/#public-version-identifiers
    version_git = tokens[0]
else:
    raise ValueError('Unexpected "git describe" output:'
                     '{}'.format(version_buf))

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
    ),  # TODO

    install_requires=[
        'six>=1.10',
        'nibabel>=2.1.0',
        'numpy>=1.11',
        'scipy>=0.18',
        'configparser',
        'tensorflow-gpu==1.1',
        'pillow',
        ],

    entry_points={
        'console_scripts': [
            'net_segmentation=niftynet:main',
        ],
    },
)

