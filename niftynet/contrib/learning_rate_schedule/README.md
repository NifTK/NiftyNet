# Learning rate decaying example

This application implements a simple learning rate schedule of
"halving the learning rate every 3 iterations" for segmentation applications.

The concept is general and could be used for other types of application.

The core function is implemented by

1) add a `self.learning_rate` placeholder, and connect it to the network
in `connect_data_and_network` function

2) add a `self.current_lr` variable to keep track of the current learning rate

3) override the default `set_iteration_update` function provided in `BaseApplication`
so that `self.current_lr` is changed according to the `current_iter`.

4) To feed the `self.current_lr` value to the network, the data feeding dictionary
is updated within the customised `set_iteration_update` function, by
```
iteration_message.data_feed_dict[self.learning_rate] = self.current_lr
```
`iteration_message.data_feed_dict` will be used in 
`tf.Session.run(..., feed_dict=iteration_message.data_feed_dict)` by the engine
at each iteration.


To run this application from command line with a config file (e.g., `config/default_segmentation.ini`):
```
python net_run.py train -a niftynet.contrib.learning_rate_schedule.decay_lr_application.DecayLearningRateApplication 
                        -c config/default_segmentation.ini 
```