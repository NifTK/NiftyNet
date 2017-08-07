#!/usr/bin/env bash

# create a virtual env to test pip installer
venv="niftynet-pip-installer-venv-py2"
mypython=$(which python2)
virtualenv -p $mypython $venv
cd $venv
source bin/activate
# print Python version to CI output
which python
python --version
# install using built NiftyNet wheel
pip install $niftynet_wheel
# install SimpleITK for package importer test to work properly
pip install simpleitk
# check whether all packages are importable
cat $package_importer
python $package_importer
# test niftynet command
ln -s /home/gitlab-runner/environments/niftynet/data/example_volumes ./example_volumes
net_segmentation train -c $niftynet_dir/config/default_config.ini --net_name toynet --image_size 42 --label_size 42 --batch_size 1 --save_every_n 10
net_segmentation inference -c $niftynet_dir/config/default_config.ini --net_name toynet --image_size 80 --label_size 80 --batch_size 8
# deactivate virtual environment
deactivate
cd $niftynet_dir
