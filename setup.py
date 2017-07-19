from setuptools import setup, find_packages
from codecs import open


# Get the summary
description = 'A convolutional neural networks platform' +\
              ' for research in medical image analysis and' +\
              ' computer-assisted intervention.'

# Get the long description
with open('pip/long_description.rst', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='NiftyNet',

    version='0.1rc1',  # TODO

    description=description,
    long_description=long_description,

    url='http://niftynet.io/',

    author='NiftyNet Consortium',
    author_email='nifty-net@live.ucl.ac.uk',

    license='License :: OSI Approved :: Apache Software License',

    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',

        'Programming Language :: Python',

        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],  # TODO

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
        'scikit-image>0.12',
        'tensorflow-gpu==1.1',
        ],

    entry_points={
        'console_scripts': [
            'niftynet=niftynet:main',  # TODO
        ],
    },
)

