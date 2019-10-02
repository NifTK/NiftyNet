# Using NiftyNet Applications as Modules in 3rd-Party Code

This simple demo shows the use of NiftyNet applications in 3rd-party code via niftynet.application.application_module_wrapper.ApplicationModuleWrapper.
The code accompanying its brief explanation is contained in `module_app.py` that builds on the unet_histology demo. The input images are assumed to be RGB bitmaps while the label images are assumed to be binary and supplied in nii.gz format.


## Model Configuration

Models driven through a app-module are configured via ini files, the same as when using `net_run`. The only difference in terms of configuration format is that file-path specification for input data is no longer required, i.e., `path_to_search`, `filename_contains`, etc.


## I/O Setup

When using a NiftyNet application as a module, input and output of images is done in-memory and asynchronously via callback functions. These callback functions are installed via the `set_input_callback` method, in the case of input images, and the `set_output_callback` method, in the case of the output of inferred images.
Input callbacks essentially replace the input file-path specifications, so one callback for every image type used by the application must be provided. In addition to these callback functions, the total number of subjects in the data set must be specified through the `set_num_subjects` method.


## Running Applications

Before an app can be executed, the type of action (train, infer, evaluate), must be specified through the `set_action` method, and the application initialised via `initialise_application`. Finally, the actual execution is done through the `run` method.
