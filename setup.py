from setuptools import setup, find_packages
from codecs import open


# Get the summary
with open('pip/description.rst', encoding='utf-8') as f:
    description = f.read()

# Get the long description
with open('pip/long_description.rst', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='NiftyNet',

    version='0.1rc1',  # TODO

    description=description,
    long_description=long_description,

    url='http://niftynet.io/',

    author='Wenqi Li, Carole Sudre, Zach Eaton-Rosen,'
           ' Jorge Cardoso, Tom Vercauteren and other'
           ' NiftyNet Contributors',
    author_email='wenqi.li@ucl.ac.uk',

    license='License :: OSI Approved :: Apache Software License',

    classifiers=[
        '',

        '',
    ],  # TODO

    packages=find_packages(exclude=['', '', '']),  # TODO

    install_requires=['', '', ''],  # TODO

    entry_points={
        'console_scripts': [
            '',  # TODO
        ],
    },
)

