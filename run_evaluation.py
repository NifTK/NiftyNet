import sys
import util
import parse_user_params

if __name__ == "__main__":
    args = parse_user_params.run_eval()
    if util.has_bad_inputs(args):
        sys.exit(-1)
    if args.action.lower() == 'roi':
        import compute_ROI_statistics
        compute_ROI_statistics.run(args)
    elif args.action.lower() == 'compare':
        import compare_segmentations
        compare_segmentations.run(args)
