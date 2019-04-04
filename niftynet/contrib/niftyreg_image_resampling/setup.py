from __future__ import print_function

import os
import os.path as osp
import platform
from setuptools import setup, Extension, Command
from setuptools.command.build_ext import build_ext
from shutil import which
import subprocess as sp
import sys

__CMAKE_OVERRIDE_FLAGS__ = {}


class CMakeExtension(Extension):
    def __init__(self, name):
        super(CMakeExtension, self).__init__(name, sources=[])


class CMakeOverride(Command):
    description = 'Overrides CMake variables for build'

    user_options = [('settings=', 's',
                     'CMake variable override: <KEY>:<VALUE>:<KEY>:<VALUE>...')]

    def initialize_options(self):
        self.settings = ''

    def finalize_options(self):
        pass

    def run(self):
        global __CMAKE_OVERRIDE_FLAGS__

        overrides = self.settings.split(':')
        for i in range(0, len(overrides), 2):
            print('Overriding %s with %s' % (overrides[i], overrides[i+1]))
            __CMAKE_OVERRIDE_FLAGS__[overrides[i]] = overrides[i+1]


class CMakeBuildExt(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        print('Building ' + ext.name)

        outdir = osp.abspath(osp.dirname(self.get_ext_fullpath(ext.name)))
        args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + outdir]
        if not osp.isdir(outdir):
            os.makedirs(outdir)
        args += ['-DGPU_RESAMPLING_CONFIGFILE_DIR=' + outdir]
        args += ['-DCMAKE_BUILD_TYPE=' + ('Debug' if self.debug else 'Release')]

        if platform.system() == 'Linux' \
           and any(dist in platform.dist() for dist in ('Debian', 'Ubuntu')):
            # Need to find compilers that play nice with nvcc;
            # this assumes compatible versions have been linked to
            # /PATH/TO/cuda/bin/cc and /PATH/TO/cuda/bin/c++, and
            # that they appear first on the search path.
            if not 'CMAKE_C_COMPILER' in __CMAKE_OVERRIDE_FLAGS__:
                args += ['-DCMAKE_C_COMPILER=' + which('cc')]
            if not 'CMAKE_CXX_COMPILER' in __CMAKE_OVERRIDE_FLAGS__:
                args += ['-DCMAKE_CXX_COMPILER=' + which('c++')]

        for key, val in __CMAKE_OVERRIDE_FLAGS__.items():
            args += ['-D' + key + '=' + val]

        args += [osp.join(osp.dirname(osp.abspath(__file__)),
                          'niftyreg_gpu_resampler')]

        if not osp.isdir(self.build_temp):
            os.makedirs(self.build_temp)

        print('Building in ' + str(self.build_temp)
              + ': cmake ' + ' '.join(args))
        sp.call(['cmake'] + args, cwd=self.build_temp)
        sp.call(['cmake'] + args, cwd=self.build_temp)
        sp.call(['cmake', '--build', self.build_temp])


setup(
    name='niftyreg_gpu_resampler',
    description='A NiftyNet image resampling sub-module powered by NiftyReg '
    'GPU code.',
    packages=['.'],
    ext_modules=[CMakeExtension('niftyreg_gpu_resampler')],
    cmdclass={'override': CMakeOverride,
              'build_ext': CMakeBuildExt},
    zip_safe=False,
)
