# Model zoo

With `net_download` command and [the model zoo
server](https://github.com/NifTK/NiftyNetModelZoo/blob/master/README.md),
NiftyNet provides convenient access to the shared trained/untrained networks.
Trained networks can be used directly (as part of a workflow or for performance
comparisons), fine-tuned for different data distributions (e.g. a different
hospital’s images), or used to initialize networks for other applications (i.e.
transfer learning).  Untrained networks or conceptual blocks can be used within
new networks.

The following sections introduce:
- `net_download` used to download model and data,
- the user's "niftynet home" folder as the output directory of `net_download`
(also as the default folder of NiftyNet).

## `net_download`
The command is available for both [pip-installed and source code repository
users](./installation.html) (the source code repository users should replace
`net_download` with `python net_download.py`):
```text
usage: net_download [-h] [-r] [-v] sample_id [sample_id ...]

Download NiftyNet sample data

positional arguments:
  sample_id      Identifier string(s) for the example(s) to download

optional arguments:
  -h, --help     show this help message and exit
  -r, --retry    Force data to be downloaded again
  -v, --version  show program's version number and exit
```


For a concrete example,
```
net_download highres3dnet_brain_parcellation_model_zoo
```
will automatically take the following two steps:

* download the model zoo entry specification
  `highres3dnet_brain_parcellation_model_zoo.ini` from [NiftyNet model zoo
  server](https://github.com/NifTK/NiftyNetModelZoo).

```ini
[config]
version = 1.0

[data]
local_id = OASIS
url = http://cmic.cs.ucl.ac.uk/platform/niftynetexamples/OASIS.tar.gz?dl=1
action = expand
destination = data

[weights]
local_id = highres3dnet_brain_parcellation
url = https://www.dropbox.com/s/nxg2ixs9rh1p9ri/highres3dnet_brain_parcellation_weights.tar.gz?dl=1
action = expand
destination = models

[network_inference_config]
local_id = highres3dnet_brain_parcellation
url = https://www.dropbox.com/s/r2q08q1kkd534p4/highres3dnet_brain_parcellation_config.tar.gz?dl=1
action = expand
destination = extensions
```

* parse each section of the `.ini` file, download `[data]`, `[weights]`, and
   `[network_inference_config]` respectively from the specified `url`, to
   user's `$NIFTYNET_HOME/[destination]/[local_id]` where `$NIFTYNET_HOME` is
   specified in [the following section](#global-settings).  These destination
   directories are designed for different types of data, possible values of
   `destination` are `data`, `models`, `extensions`. Specifically,

  - `data` directory stores example image inputs.
  - `models` directory stores trained model weights
  - `extensions` directory stores Python implementation of
networks, loss functions, new applications, etc.

_Depending on their availability, some model zoo entries do not contain
data, or trained weight, or both._

_See also: [submitting new model zoo entries](contributing.html#submitting-model-zoo-entries)._




## Global-settings

The global NiftyNet configuration is read from
`$niftynet_config_home/config.ini`, where `niftynet_config_home` is a system
environment variable.  NiftyNet attempts to load this file (defaulting to
`~/.niftynet/config.init` if undefined) for the global configuration.

* If it does not exist, NiftyNet will create a default one.
* If it exists but cannot be read (e.g., due to incorrect formatting):
  - NiftyNet will back it up with a unique string;
  - create a default one.

Currently the minimal version of this file will look like the following:
```ini
[global]
home = ~/niftynet
```

The `home` key specifies the root folder (referred to as `$NIFTYNET_HOME` from
this point onwards) to be used by NiftyNet for storing and locating user data
such as downloaded models, and new networks implemented by the user.  This
setting is configurable, and upon successfully loading this file NiftyNet will
attempt to create the specified folder, if it does not already exist.

On first run, NiftyNet will also attempt to create the NiftyNet extension
module hierarchy (`extensions.*`), that allows for the discovery of
user-defined networks.  This hierarchy consists of the following:

* `$NIFTYNET_HOME/extensions/` (folder)
* `$NIFTYNET_HOME/extensions/__init__.py` (file)
* `$NIFTYNET_HOME/extensions/network/` (folder)
* `$NIFTYNET_HOME/extensions/network/__init__.py` (file)

_To completely uninstall NiftyNet, please manually remove `~/.niftynet` and
`$NIFTYNET_HOME` folders._
