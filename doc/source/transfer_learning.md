# Transfer learning

With NiftyNet, it's possible to initialize a model with pre-trained weights and
adapt it to perform a different task. This functionality is provided through two
config file parameters: `vars_to_restore` and `freeze_restored_vars`.

### Setting up your model directory

To fine-tune your model on a new dataset, first create a new model directory and
specify its location through the `model_dir` parameter in your config file.
Inside your model_dir directory create a folder named *models* and place the
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
use the latest model in the *models* folder. You can freely change the `###` in
the filenames to anything you want, however setting it to 0 will create a new
model with randomized variables. Just change `###` to 1 if you want a fresh
iteration counter.

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

# ckpt_path: full path to checkpoint file (ex: /path/to/ckpt/model.ckpt-###)
# output_file: name of output file (ex: /path/to/file/net_vars.txt)
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

**For matching a few variables:**

`^.*(conv_1|conv_2).*$` = match all vars that have *conv_1* or *conv_2* in
their name (the '|' acts like a boolean OR).

**For excluding a few variables:**

`^((?!DenseVNet\/(conv|batch_norm)).)*$` = don't match any vars that contain
*DenseVNet/conv* or *DenseVNet/batch_norm* in their name (the '\' is an escape
character for '/').

Once you've created a regex expression, it's recommended that you use a tool
like [RegEx101](https://regex101.com) to double check that it works as
expected. Set the test string as the variable names returned by the
`get_ckpt_vars()` function above. If it checks out, set `vars_to_restore` to
your regex.


### Freezing model weights

If you only wish to optimise the randomized variables of your network, you can
set `freeze_restored_vars` to **True**. This is useful for only training the top
layers of a network while leaving the pre-trained feature extractors intact.

While `freeze_restored_vars` does not allow you to pick and choose which vars
to optimise, it stills offers quite a bit of flexibility. For example, when
adapting your network to a new task your can first freeze your feature
extractors until your top layers are reasonably well trained. Then you can stop training, unfreeze all layers and proceed to fine-tune the whole network
with a lower learning rate.

### *Notes*

- Transfer learning in NiftyNet only works between networks using the exact same
graph. Therefore you cannot initialize variables with weights from a random
pre-trained VGG net for example, unless they share all of the same variables
and they are named the same.
