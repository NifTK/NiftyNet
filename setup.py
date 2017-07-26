from setuptools import setup, find_packages


# Get the summary
description = 'An open-source convolutional neural networks platform' +\
              ' for research in medical image analysis and' +\
              ' image-guided therapy'

# Get the long description
with open('pip/long_description.rst') as f:
    long_description = f.read()


setup(
    name='NiftyNet',

    version='0.1rc1',  # TODO

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

