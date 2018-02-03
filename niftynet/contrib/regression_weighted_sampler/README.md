### Training regression network with weighted sampling of image windows

ref:

Berger et al., "An Adaptive Sampling Scheme to Efficiently Train Fully Convolutional Networks for Semantic Segmentation",
https://arxiv.org/pdf/1709.02764.pdf


To run this application (from NiftyNet source code root folder)
with a config file (e.g., `niftynet/contrib/regression_weighted_sampler/isampler.ini`)

```bash
model_dir="regression_model"
mkdir $model_dir
# copying the initial weight maps
cp -r /data/segmentation_mask $model_dir/error_maps
# in configuration set SAMPWEIGHT path to the value of $model_dir/error_maps
python net_run.py train -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
                        -c niftynet/contrib/regression_weighted_sampler/isampler.ini
                        --model_dir $model_dir --starting_iter 0 --max_iter 500

# generating error maps on the training data using the latest model
python net_run.py inference -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
                            -c niftynet/contrib/regression_weighted_sampler/isampler.ini
                            --inference_iter -1 --error_map True --batch_size 1 --dataset_split_file nofile

# continue training from the latest model
python net_run.py train -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
                        -c niftynet/contrib/regression_weighted_sampler/isampler.ini
                        --model_dir $model_dir --starting_iter -1 --max_iter 200
# ...
```


To do the train/inference in Bash:
```bash
label_data_dir="/mydata/segmentation_mask"
model_dir="regression_isampler"
mkdir $model_dir
cp -r $label_data_dir $model_dir/error_maps
python net_run.py train -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
                        -c isampler_config.ini --model_dir $model_dir --starting_iter 0 --max_iter 500

for iter in `seq 500 500 10000`;
do
  python net_run.py inference \
    -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
    -c isampler_config.ini --inference_iter -1 --error_map True --batch_size 2 --dataset_split_file nofile

  python net_run.py train \
    -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
    -c isampler_config.ini --starting_iter -1 --max_iter $iter --model_dir $model_dir
done
```

To do an "autocontext" training:
```bash
model_dir="autocontext_regression"
initial_mask="/mydata/segmentation_mask"
mkdir $model_dir;
cp -r $initial_mask $model_dir/error_maps;
python net_run.py train -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
                        -c net_autocontext.ini --starting_iter 0 --max_iter 10

for i in `seq 1000 1000 10000`;
do
  python net_run.py inference \
    -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
    -c net_autocontext.ini --inference_iter -1 --error_map True --batch_size 2 --dataset_split_file nofile

  python net_run.py train \
    -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
    -c net_autocontext.ini --starting_iter -1 --max_iter $i

done
```

To do an "autocontext" inference
```bash
model_dir="autocontext_regression"
initial_mask="/mydata/segmentation_mask"
rm -r $model_dir/error_maps;
cp -r $initial_mask $model_dir/error_maps

python net_run.py inference \
  -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
  -c ~/from_felix/net_auto.ini --inference_iter 10 --batch_size 5 --error_map True --dataset_split_file nofile

for i in `seq 1000 1000 9000`;
do
  python net_run.py inference \
    -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
    -c ~/from_felix/net_auto.ini --inference_iter $i --error_map True --batch_size 5 --dataset_split_file nofile
done
python net_run.py inference \
  -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
  -c ~/from_felix/net_auto.ini --inference_iter 10000 --error_map False --batch_size 5 --dataset_split_file nofile
```


### Parameters for config file in the regression section
|Params.| Type |Example|Default|
|---|---|---|---|
|[error_map](#error_map)|Boolean|`error_map=True`|`'False'`|


###### `error_map`
At inference time, setting the parameter to True to generate errormaps on the input data.

