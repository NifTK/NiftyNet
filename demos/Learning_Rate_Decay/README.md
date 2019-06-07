# Learning rate decay application

This application implements a simple learning rate schedule of
"halving the learning rate every 3 iterations" for segmentation applications.

The concept is general and could be used for other types of application. A brief demo is provide which can be fully run from a jupyter notebook provided a a working installation of NiftyNet exists on your system.

The core function is implemented by:

1) Adding a `self.learning_rate` placeholder, and connect it to the network
in `connect_data_and_network` function

2) Adding a `self.current_lr` variable to keep track of the current learning rate

3) Overriding the default `set_iteration_update` function provided in `BaseApplication`
so that `self.current_lr` is changed according to the `current_iter`.

4) To feed the `self.current_lr` value to the network, the data feeding dictionary
is updated within the customised `set_iteration_update` function, by
```
iteration_message.data_feed_dict[self.learning_rate] = self.current_lr
```
`iteration_message.data_feed_dict` will be used in 
`tf.Session.run(..., feed_dict=iteration_message.data_feed_dict)` by the engine
at each iteration.


*This demo only supports NiftyNet cloned from [GitHub](https://github.com/NifTK/NiftyNet).*
Further demos/ trained models can be found at [NiftyNet model zoo](https://github.com/NifTK/NiftyNetModelZoo/blob/master/dense_vnet_abdominal_ct_model_zoo.md).
