from itertools import product

import itertools


def dict_product(dicts):
    """
    from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def main():
    base_command = "python net_segment.py train -c ./demos/unet/U373.ini"
    # note: these paths are relative to the model directory
    d_split_files = ["../../demos/unet/d_split_%i.csv" % i for i in [1, 2]]
    conditions = {"--do_elastic_deformation": ["True", "False"],
                  "--random_flipping_axes": ["'0,1'", "-1"],
                  "--dataset_split_file": d_split_files}

    combinations = list(dict_product(conditions))
    for combo in combinations:
        str_command = [base_command]
        for condition in combo:
            str_command += [str(condition), combo[condition]]
        print(' '.join(str_command))


if __name__ == "__main__":
    main()
