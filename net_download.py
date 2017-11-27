#!/usr/bin/env python
#  -*- coding: utf-8 -*-
import sys

from niftynet.utilities.download import main
from niftynet.utilities.niftynet_global_config import NiftyNetGlobalConfig

if __name__ == "__main__":
    # make sure home folder is initialised
    NiftyNetGlobalConfig()
    sys.exit(main())
