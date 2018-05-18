# Freehand Ultrasound Image Simulation with Spatially-Conditioned Generative Adversarial Networks

This page describes how to acquire and use the network described in

Yipeng Hu, Eli Gibson, Li-Lin Lee, Weidi Xie, Dean C. Barratt, Tom Vercauteren, J. Alison Noble
(2017). [Freehand Ultrasound Image Simulation with Spatially-Conditioned Generative Adversarial Networks](https://arxiv.org/abs/1707.05392), In MICCAI RAMBO 2017

## Downloading model zoo file and conditioning data

The model zoo file is available [here](https://www.dropbox.com/s/etptck5yi1fzvkr/ultrasound_simulator_gan_model_zoo.tar.gz?dl=0).  Extract the model directory named `ultrasound_simulator_gan_model_zoo` from this file.

This network generates ultrasound images conditioned by a coordinate map. Example coordinate maps are available [here](https://www.dropbox.com/s/w0frdlxaie3mndg/test_data.tar.gz?dl=0)). Extract `test_data.tar.gz` to a data directory.

## Editing the configuration file

Inside the model_dir is a configuration file: `inference_config.ini`. You will need to change this file to point to the correct paths for the model directory (`model_dir=`) and the conditioning data (`path_to_search=`).

## Generating samples

Generate samples from the simulator with the command `net_gan.py inference -c ultrasound_simulator_gan_model_zoo/inference_config.ini`


