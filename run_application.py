# -*- coding: utf-8 -*-
import sys
import time
import warnings

from niftynet import main

warnings.simplefilter('always', DeprecationWarning)
warnings.warn('run_application.py is deprecated and will be removed; '
              'please use net_segment.py instead.',
              DeprecationWarning, stacklevel=2)
warnings.simplefilter('ignore', DeprecationWarning)
time.sleep(3)

if __name__ == "__main__":
    sys.exit(main())
