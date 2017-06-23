#!/usr/bin/env sh 

# please make sure you installed all dependencies of NiftyNet.
# cd NiftyNet/; pip install -r requirements-gpu.txt
NIFTYNET=../../run_application.py
GPU_CHECKER=../../utilities/check_gpu.py

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

# download OASIS dataset ~7MB
sh ../../data/brain_parcellation/get_oasis_data.sh

# download a trained HighRes3DNet parameters
sh get_highres3dnet_model.sh

# find a large GPU
export CUDA_VISIBLE_DEVICES=$(python -u $GPU_CHECKER 10000)
echo "using GPU: $CUDA_VISIBLE_DEVICES"

python -u $NIFTYNET inference -c models/highres3dnet_config_eval.txt --image_size 160 --label_size 160

# please see the output at demos/brain_parcellation/results
