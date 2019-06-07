# Finetuning networks

With NiftyNet, it's possible to initialize your neural net with pre-trained
variables and then fine-tune it for a seperate but similar task. This
functionality is provided through two config file parameters: `vars_to_restore`
and `vars_to_freeze`.

### Setting up your model directory

To fine-tune your model on a new dataset, first create a new model directory and
specify its location through the `model_dir` parameter in your config file.
Inside your `model_dir` directory create a folder named `models` and place the
three checkpoint files which constitue your model inside. Your directory
structure should look like this:

```
model_dir/
  models/
    model.ckpt-###.data-00000-of-00001
    model.ckpt-###.index
    model.ckpt-###.meta
```

You can specify which checkpoint sources the pre-trained variables by
setting the `starting_iter` parameter in your config file. The `starting_iter`
should be set to either the `###` in your checkpoint filenames or -1, which will
use the latest model in the `models` folder. You can freely change `###` in
the filenames to anything you want, however specifying 0 will cause NiftyNet
to ignore existing models and create a new model with randomized variables.
Just change `###` to 1 if you want a fresh iteration counter.

### Selecting variables to restore

Next we must decide which variables we would like to restore. The
`vars_to_restore` parameter allows you to specify a regular expression that
will match variable names in a checkpoint file. Only variable names matched by
the regex will be restored by NiftyNet while the rest are initialized to
random values.

You can obtain a list of all the variables in your model using the following
bit of code:

```python
import tensorflow as tf

# ckpt_path: full path to checkpoint file (e.g.: /path/to/ckpt/model.ckpt-###)
# output_file: name of output file (e.g.: /path/to/file/net_vars.txt)
def get_ckpt_vars(ckpt_path, output_file):
    file = open(output_file, 'w+')
    for var in tf.train.list_variables(ckpt_path):
        file.write(str(var) + '\n')
    file.close()

get_ckpt_vars('~/Desktop/model_outputs/models/model.ckpt-1', \
              '~/Desktop/net_vars.txt')
```

Once you've determined which variables you plan to restore, you must write a
regex which will match them. If you have little experience with regex, here are
a few examples to get you started:

**For matching variables:**

`vars_to_restore=^.*(conv_1|conv_2).*$` will match all trainable variables that
have `conv_1` or `conv_2` in their name (the `|` acts like a boolean OR).

**For excluding variables:**

`vars_to_restore=^((?!DenseVNet\/(skip_conv|fin_conv)).)*$` will not match any
trainable variables that contain
`DenseVNet/skip_conv` or `DenseVNet/fin_conv` in their name (the `\` is an
escape character for `/`).

Once you've created a regex expression, it's recommended that you use a tool
like [RegEx101](https://regex101.com) to double check that it works as
expected. Set the test string as the list of variable names returned by
`get_ckpt_vars()`. If the regex successfuly selects the lines you intended, then
you can use it to set `vars_to_restore`.


### Freezing model weights
The model variables matched by `vars_to_restore` will be restored from
checkpoint and by default, remain intact during training.  To change the model
parameter updating behaviour, you can also specify a regular expression of
`vars_to_freeze`.  The matched variables within the collection of trainable
parameters (`tf.trainable_variables()`) will *not* be updated during training.
This is useful for training a subset of layers of a network while leaving the
rest of the network fixed.


### Common Pitfalls

Transfer learning in NiftyNet will work between any models that share the same
variables as those being restored. In other words, you can completely change
network layers that you plan to randomise but if you try to restore variables
that aren't the exact shape and name between models, Tensorflow and NiftyNet
will throw an error. For example, you may encounter the following error in your
training log:

```
CRITICAL:niftynet:2018-10-24 00:53:22,438: checkpoint ~/outputs/models/
model.ckpt-### not found or variables to restore do not match the current
application graph
```

This means that certain variables in your network are not present in your model
checkpoint. Often this can occur unexpectedly when you restore variables that
were frozen during the previous round of training. Since these variables
weren't being trained, optimizer specific variables used in methods like Adam
were never created and therefore never saved in the checkpoint. You can
overcome this by simply restoring all variables except for those used by Adam:
`vars_to_restore = ^((?!(Adam)).)*$`. In general, if you read the error thrown
by Tensorflow, you should be able to figure out which variables are causing the
problem.
