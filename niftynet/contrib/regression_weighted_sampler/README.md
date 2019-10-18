### Training regression network with weighted sampling of image windows

ref:

Berger et al., "An Adaptive Sampling Scheme to Efficiently Train Fully Convolutional Networks for Semantic Segmentation",
https://arxiv.org/pdf/1709.02764.pdf

Please see [the model zoo entry](https://github.com/NifTK/NiftyNetModelZoo/blob/master/mr_ct_regression_model_zoo.md)
for instructions for downloading data and config file.

### Training regression network with autocontext model

Please see [the model zoo entry](https://github.com/NifTK/NiftyNetModelZoo/blob/master/autocontext_mr_ct_model_zoo.md)
for instructions for downloading data and config file.


### Parameters for config file in the regression section
|Params.| Type |Example|Default|
|---|---|---|---|
|[error_map](#error_map)|Boolean|`error_map=True`|`'False'`|


###### `error_map`
At inference time, setting the parameter to True to generate errormaps on the input data.

