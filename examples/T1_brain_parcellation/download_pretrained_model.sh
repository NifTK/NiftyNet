#!/usr/bin/env sh
# Download model and unzip

cd downloaded
mkdir highres3dnet_model
mkdir highres3dnet_model/models
rsync emerald:/home/ucl/eisuc269/replicate/models/model_highres3dnet/models/model.ckpt-13000\* ./highres3dnet_model/models

cd ../../
python -u run_application.py inference -c examples/T1_brain_parcellation/downloaded/highres3dnet_model/highres3dnet_config_eval.txt --image_size 120 --label_size 120
