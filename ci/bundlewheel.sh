#!/usr/bin/env bash

python setup.py bdist_wheel

export niftynet_wheel=$(ls $niftynet_dir/dist/*.whl)  # there will be only one file!

