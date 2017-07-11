from __future__ import absolute_import, print_function
import sys

import niftynet.utilities.misc_common as util
import niftynet.utilities.parse_user_params as parse_user_params

if __name__ == "__main__":
    args, csv_dict = parse_user_params.run_eval()
    if util.has_bad_inputs(args):
        sys.exit(-1)
    if args.action.lower() == 'roi':
        import niftynet.evaluation.compute_ROI_statistics

        niftynet.evaluation.compute_ROI_statistics.run(args, csv_dict)
    elif args.action.lower() == 'compare':
        import niftynet.evaluation.compare_segmentations

        niftynet.evaluation.compare_segmentations.run(args, csv_dict)
