from setuptools import setup, find_packages
from codecs import open


# Get the summary
description = 'NiftyNet is an open-source platform for convolutional' +\
              ' neural networks for research in medical image' +\
              ' analysis and computer-assisted intervention.'

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
    author_email='wenqi.li@ucl.ac.uk',

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

    install_requires=['', '', ''],  # TODO

    entry_points={
        'console_scripts': [
            '',  # TODO
        ],
    },
)

