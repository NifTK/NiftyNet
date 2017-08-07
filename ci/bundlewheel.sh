#!/usr/bin/env bash

# remove existing installers
rm -f $niftynet_dir/dist/*.whl

# bundle installer
python setup.py bdist_wheel

# inform other scripts of wheel's location
export niftynet_wheel=$(ls $niftynet_dir/dist/*.whl)  # there will be only one file!
