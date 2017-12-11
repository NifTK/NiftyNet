# Model zoo

## Global-settings

The global NiftyNet configuration is read from `~/.niftynet/config.ini`.
When NiftyNet is run, it will attempt to load this file for the global configuration.
* If it does not exist, NiftyNet will create a default one.
* If it exists but cannot be read (e.g., due to incorrect formatting):
- NiftyNet will back it up with a timestamp (for instance
  `~/.niftynet/config-backup-2017-10-16-10-50-58-abc.ini` - `abc` being a
  random string) and,
- Create a default one.
* Otherwise NiftyNet will read the global configuration from this file.

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
module hierarchy (`niftynetext.*`), that allows for the discovery of
user-defined networks.  This hierarchy consists of the following:

* `$NIFTYNET_HOME/niftynetext/` (folder)
* `$NIFTYNET_HOME/niftynetext/__init__.py` (file)
* `$NIFTYNET_HOME/niftynetext/network/` (folder)
* `$NIFTYNET_HOME/niftynetext/network/__init__.py` (file)

Alternatively this hierarchy can be created by the user before running NiftyNet
for the first time, e.g. for [defining new networks][new-network].

[new-network]: ../niftynet/network/README.md
