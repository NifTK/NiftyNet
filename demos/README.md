### Usage

##### (a) Running the demos:
Please see the `README.md` in each folder of this [directory](./demos) for more details.


##### (b) To run an application 

For example, to run a segmentation network:
``` sh
# training
net_segment train -c /path/to/customised_config
# inference
net_segment inference -c /path/to/customised_config
```
where `/path/to/customised_config` implements all parameters listed by running:
```sh
net_segment -h
```
please see [configuration](../config) documentations for more details and 
config file examples.

Commandline parameters override the default settings defined in `/path/to/customised_config`.
For example,
``` sh
# training
net_segment train -c /path/to/customised_config --lr 0.1
```
Uses all parameter specified in `/path/to/customised_config` but sets the
learning rate to `0.1`
