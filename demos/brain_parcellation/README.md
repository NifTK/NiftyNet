## To run the brain parcellation demo


Run `sh run_parcellation.sh`

The script will:

1) **Download a MR volume (originally from [OASIS](http://www.oasis-brains.org/) dataset; about 7MB)**

This step will create an `OASIS_DATA.tar.gz` file and an `OASIS` folder in `NiftyNet/data/parcellation/`.

2) **Download a trained model of HighRes3DNet (about 64MB)**

This step will create a `highres3dnet.tar.gz` file and a `models` folder in `NiftyNet/demo/parcellation/`.

3) **Run NiftyNet inference program**

This step will create a `image_files.csv` file in `NiftyNet/demo/parcellation/`.
The parcellation output will be stored in `NiftyNet/demo/parcellation/results`.

_Please Note:_

* This demo requires an GPU with at least 10GB memory.

* Please change the environment variable `CUDA_VISIBLE_DEVICES` to an appropriate value if necessary. (e.g. `export CUDA_VISIBLE_DEVICES=0` will allow NiftyNet to use the `0` GPU)