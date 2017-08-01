# -*- coding: utf-8 -*-
import warnings
import time
warnings.simplefilter('always', DeprecationWarning)
warnings.warn('run_application.py is deprecated and will be removed; please use net_segmentation.py instead.', DeprecationWarning, stacklevel=2)
warnings.simplefilter('ignore', DeprecationWarning)
time.sleep(3)

from niftynet import main
import sys

if __name__ == "__main__":
    sys.exit(main())

