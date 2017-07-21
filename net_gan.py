# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import os
import sys

import niftynet.utilities.misc_common as util
import niftynet.utilities.parse_user_params_gan as parse_user_params
from niftynet.engine.application_driver import ApplicationDriver


def main():
    param, csv_dict = parse_user_params.run()
    if util.has_bad_inputs(param):
        return -1
    else:
        # writing user configurations for future reference
        settings_filename = os.path.join(param.model_dir,
                                         'settings_' + param.action + '.txt')
        util.print_save_input_parameters(param, txt_file=settings_filename)

    app_driver = ApplicationDriver()
    app_driver.initialise_application(csv_dict, param)
    app_driver.run_application()
    return 0


if __name__ == "__main__":
    sys.exit(main())
