import os
import sys
import src.parse_user_params  as parse_user_params
import src.util as util
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)


if __name__ == "__main__":
    args = parse_user_params.run_eval()
    if util.has_bad_inputs(args):
        sys.exit(-1)
    import run.evaluation as evaluation
    evaluation.run(args)
