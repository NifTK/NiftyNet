#!/usr/bin/env python

import numpy as np
import h5py as h5
import argparse
import glob
import fnmatch
import re
import os

import imageio
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig

global_config = NiftyNetGlobalConfig()

_desc = "Transforms images to an HDF5 format readable by NiftyNet"

parser = argparse.ArgumentParser(description=_desc)
parser.add_argument('pattern', nargs='+',
                    help='One or more multiple data patterns.')
parser.add_argument('collection', type=str,
                    help='Collection name, required to store the data.')
parser.add_argument('-n', '--name', dest='dname', type=str, default='data',
                    help='Dataset name, required to store the data.')
parser.add_argument('-r', '--recursive', action="store_true",
                    help='Wether to recursively look for data into directories'
                         '(default: False).')
parser.add_argument('-f', '--force', dest='overwrite', action="store_true",
                    help='Overwrite HDF5 file (default: False).')
parser.add_argument('-d', '--dtype', dest='dtype', default=None,
                    help='Transform image data to a different data type')


args = parser.parse_args()

all_files = []
for pattern in args.pattern:
    regexp = pattern.replace('*', '[^0-9]*([0-9]+)[^0-9]*')
    filenames = glob.glob(pattern, recursive=args.recursive)
    for fname in filenames:
        found_idx = re.findall(regexp, fname)
        if found_idx:
            all_files.append((int(found_idx[0]), fname))

home_folder = global_config.get_niftynet_home_folder()
data_folder = os.path.join(home_folder, 'data', args.collection)

if not os.path.isdir(data_folder):
    os.makedirs(data_folder)

data_file = os.path.join(data_folder, args.dname + '.h5')
if os.path.isfile(data_file) and not args.overwrite:
    raise ValueError('Dataset {} already exists.'.format(data_file))

with h5.File(data_file, 'w') as f:
    for idx, filename in all_files:
        data = imageio.imread(filename)
        if args.dtype:
            data = data.astype(np.dtype(args.dtype))
        f.create_dataset('subject_{:08d}'.format(idx), data=data)
