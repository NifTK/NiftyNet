import sys
import utilities.misc as util
import utilities.parse_user_params as parse_user_params

if __name__ == "__main__":
    args = parse_user_params.run_eval()
    if util.has_bad_inputs(args):
        sys.exit(-1)
    if args.action.lower() == 'roi':
        import evaluation.compute_ROI_statistics
        evaluation.compute_ROI_statistics.run(args)
    elif args.action.lower() == 'compare':
        import evaluation.compare_segmentations
        evaluation.compare_segmentations.run(args)
