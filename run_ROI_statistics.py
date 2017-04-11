import sys
import parse_user_params
import util


if __name__ == "__main__":
    args = parse_user_params.run_stats()
    if util.has_bad_inputs(args):
        sys.exit(-1)
    import compute_statistics
    compute_statistics.run(args)
