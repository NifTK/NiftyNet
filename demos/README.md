### Usage

##### (a) Running the demos:
Please see the `README.md` in each folder of this [directory](./demos) for more details.


##### (b) To run an application 

For example, to run a segmentation network <sup>(*)</sup>:
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

 <sup>(*) Please note that these instructions are for a `pip`-installed NiftyNet.
If you are using the NiftyNet command line interface from within the NiftyNet source code, please use `python net_segment.py [...]` (provided that the current working directory is the root folder of the NiftyNet repository clone) instead of `net_segment [...]`.
</sup>
