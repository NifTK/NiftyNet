# Explanation of the config file

You can amend the config files for your own purposes to run your experiments. It enables a greater flexibility in your choices of networks, losses, datasets and replace your command line arguments

## Contraints on filenames
In order to specify the files to use and the modalities to gather, you can specify them using the following conventions

For instance, an input can be specified as:

[image modality #i]  
path_to _ search = Filepath to the directory of choice  
filename_ contain = string that has to be in the filename  
filename_not _contain = string that should not be present in the filename  

an output image is specified as:  

[label modality #i]  
path_to_search = Filepath to the directory of choice  
filename_contain = string that has to be in the filename
filename_not_contain = string that should not be present in the filename  

optionally you may want to include additional weights images

[weight modality #i]  
path_to_search = Filepath to the directory of choice  
filename_contain = string that has to be in the filename
filename_not_contain = string that should not be present in the filename 

The keywords are **image**, **label** and **weight**

## Network settings
[settings]
### Histogram normalisation
The histogram normalisation is performed using the method described. The folllowing fields can be specified:  

* histogram _ ref _ file: Name of the file that contains the normalisation parameter if it has been trained before or where to save it
* norm _ type: type of landmarks used in the histogram for the matching (percentile or quartile)
* cutoff _ min: inferior cutoff of the histograms for the matching
* cutoff _ max: superior cutoff of the histogram for the matching
* multimod _ mask _ type: At training time, logical strategies to apply between the masks obtained for multiple modalities: can take one of the following:
	* or = union of the available masks
	* and = intersection of the available masks
	* all = a different mask is applied for each modality
* mask _ type = strategy applied to mask an image. Choice between:
	* otsu _ plus
	* otsu _ minus
	* thresh _ plus
	* thresh _ minus  	

### Volume preprocessing
List of different operations that can be applied to the data:  

* reorientation = [True/False] Indicates if the images should be reoriented to the standard R.A.S convention
* resampling = [True/False] Indicates if the images should be made isotropic [1,1,1]
* normalisation = [True/False] Indicates if an histogram standardisation should be applied to the data
* whitening = [True/False] Indicates if the loaded image should be whitened I->(I-mean)/std
* image _ interp _ order = [0/3] Interpolation order applied on the image. 0 for nearest neighbour, 3 for cubic interpolation
* label _interp _ order = [0/3] Idem but for the labels interpolation
* w_map _ interp_order = 3 Idem but for the additional weights
* queue_length = 3
* num_threads = 1
* num_gpus = 1
* sample_per _volume = 32

### Augmentation at training
* rotation = [True/False] Indicates if a random rotation operation should be applied to the volumes
rotation = False
* min_angle = -10.0 Minimum angle of rotation
* max_angle = 10.0 Maximum angle of rotation
* spatial_scaling = [True/False] Indicates if a random spatial scaling should be applied
* max_percentage = 10.0 Maximum zooming applied

### Network parameters
* spatial_rank = [2/2.5/3]. Dimensionality of the network input. 2.5 refers to the treatment slice by slice of a 3D image
* batch_size = 4
* image_size = 63
* label_size = 63. ***WARNING*** The label size and image size are network dependent. Details can be found in the network documentation
* w_map _size = 63
* num_classes = 5 In case of a classification task, number of possible labels 
* cuda_devices = " "
* model_ dir = ./path_ to_model Folder where to save/load the model parameters
* net_name = toynet Name of the model to use (one of those defined in the corresponding folder 

### Training specific parameters
* lr = 0.01. Learning rate to be applied
* decay = 1e-7
* loss_type = Dice. Loss function to be used
* reg_type = L2. Regularisor to be used 
* starting_iter = 0. Starting iteration
* save_every _n = 100. Frequency of model saving
* max_iter = 100. Maximum number of training steps
* volume_padding _size = 21 One side length of the receptive field affected by the network

### Inference specific parameters
* border = 0. Size of the non predicted border of the image
* pred_iter = 10. Model iteration to reload for the evaluation
* save_seg _ dir = ./path_to_save . Path to the folder where to save the output of the inference 
* eval_data _ dir = ./path_ to_test . Path to the folder where the testing data is located
* output_interp _order = 3
* output_prob = False










