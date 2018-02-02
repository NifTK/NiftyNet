### Training regression network with weighted sampling of image windows

To run this application (from NiftyNet source code root folder)
with a config file (e.g., `niftynet/contrib/regression_weighted_sampler/isampler.ini`)
```bash
cp -r /data/segmentation_mask/ model_dir/error_maps
python net_run.py train -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
                        -c niftynet/contrib/regression_weighted_sampler/isampler.ini

python net_run.py inference -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
                            -c niftynet/contrib/regression_weighted_sampler/isampler.ini
```

```bash
$model_dir='regression_model'
mkdir $model_dir
# copying the initial weight maps
cp -r /data/segmentation_mask $model_dir/error_maps
python net_run.py train -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
                        -c niftynet/contrib/regression_weighted_sampler/isampler.ini
                        --model_dir $model_dir
# generating error maps on the training data
python net_run.py inference -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
                            -c niftynet/contrib/regression_weighted_sampler/isampler.ini
                            --inference_iter -1 --error_map True
# continue training
python net_run.py train -a niftynet.contrib.regression_weighted_sampler.isample_regression.ISampleRegression \
                        -c niftynet/contrib/regression_weighted_sampler/isampler.ini
                        --model_dir $model_dir --starting_iter -1
```

### Parameters for config file in the regression section
|Params.| Type |Example|Default|
|---|---|---|---|
|[rand_samples](#rand_samples)|Integer|`rand_samples=5`|`'0'`|
|[min_numb_labels](#min_numb_labels)|Integer|`min_numb_labels=3`|`'1'`|
|[proba_connect](#proba_connect)|Boolean|`proba_connect=False`|`'False'`|
|[min_sampling_ratio](#min_sampling_ratio)|Float|`min_sampling_ratio=0.001`|`'0.00001'`|
|[compulsory_labels](#compulsory_labels)|Integer array|`compulsory_labels=0, 2`|`0, 1`|


###### `rand_samples`
Integer - Number of samples taken without any sampling rule / using a uniform sampler

###### `min_numb_labels`
Integer - Minimum number of labels present in a sampling window

###### `proba_connect`
Boolean - Indicates the distribution of sizes of connected components should be considered in the sampling

###### `min_sampling_ratio`
Float - Minimum frequency of each label in the sampled window

###### `compulsory_labels`
List of integers - Labels that must be present in the sampled window
