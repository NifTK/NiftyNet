from setuptools import setup, find_packages


setup(
    name='NiftyNet',

    version='0.1rc1',  # TODO

    description=open('pip/description.rst').read(),
    long_description=open('pip/long_description.rst').read(),

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

    keywords='',  # TODO

    packages=find_packages(exclude=['', '', '']),  # TODO

    install_requires=['', '', ''],  # TODO

    entry_points={
        'console_scripts': [
            '',  # TODO
        ],
    },
)

