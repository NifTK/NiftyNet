import os
import sys
import parse_user_params
import util
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)


if __name__ == "__main__":
    args = parse_user_params.run_stats()
    if util.has_bad_inputs(args):
        sys.exit(-1)
    import statistics
    statistics.run(args)
