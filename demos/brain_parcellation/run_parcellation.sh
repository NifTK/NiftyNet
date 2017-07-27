#!/usr/bin/env sh 

# please make sure you installed all dependencies of NiftyNet.
# cd NiftyNet/; pip install -r requirements-gpu.txt
NIFTYNET=../../net_segmentation.py

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# download OASIS dataset ~7MB
sh ../../data/brain_parcellation/get_oasis_data.sh

# download a trained HighRes3DNet parameters
sh get_highres3dnet_model.sh

# run brain parcellation
python -u $NIFTYNET inference -c ./highres3dnet_config_eval.ini --image_size 160 --label_size 160

