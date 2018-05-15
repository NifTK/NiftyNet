import itertools


class ExperimentProtocol(object):
    def __init__(self, base_command, conditions, model_dir_prefix):
        """
        :param base_command: the base command used for NiftyNet
        :param conditions: dictionary of experiment conditions,
        {"name" : {"--command_line_command": "values (iterable)"} }
        e.g. {"elastic": {"--do_elastic_deformation": ["True", "False"]}}
        :param model_dir_prefix: prefix for the experimental directory
        """
        self.base_command = base_command
        self.conditions = conditions
        self.model_dir_prefix = model_dir_prefix
        self.commands = []

    def add_condition(self, new_condition):
        for cond in new_condition:
            self.conditions[cond] = new_condition[cond]

    def to_file(self, file_name):
        with open(file_name, 'w') as f:
            for line in self.commands:
                f.write(line + "\n")

    def generate_commands(self):
        self.commands = []
        combinations = list(dict_product(self.conditions))
        for i, combo in enumerate(combinations):
            str_command = [self.base_command]
            for condition in combo:
                str_command += [str(condition), combo[condition]]
            str_command += ["--model_dir",
                            self.model_dir_prefix + '_' + str(i).zfill(len(combinations) // 10 + 1)]
            self.commands += [" ".join(str_command)]

    def __str__(self):
        return "\n".join(self.commands)


def dict_product(dicts):
    """
    from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def main():
    base_command = "python net_segment.py train -c ./demos/unet/U373.ini"
    model_dir_prefix = "./models/U373"
    # note: these paths are relative to the model directory
    d_split_files = ["../../demos/unet/u373_d_split_%i.csv" % i for i in [1, 2]]
    conditions = {"--do_elastic_deformation": ["True", "False"],
                  "--random_flipping_axes": ["'0,1'", "-1"],
                  "--dataset_split_file": d_split_files}

    u373_experiments = ExperimentProtocol(base_command, conditions, model_dir_prefix=model_dir_prefix)
    u373_experiments.generate_commands()
    u373_experiments.to_file("./run_U373.sh")

    base_command = "python net_segment.py train -c ./demos/unet/HeLa.ini"
    model_dir_prefix = "./models/HeLa"
    # note: these paths are relative to the model directory
    d_split_files = ["../../demos/unet/hela_d_split_%i.csv" % i for i in [1, 2]]
    conditions = {"--do_elastic_deformation": ["True", "False"],
                  "--random_flipping_axes": ["'0,1'", "-1"],
                  "--dataset_split_file": d_split_files}

    hela_experiments = ExperimentProtocol(base_command, conditions, model_dir_prefix=model_dir_prefix)
    hela_experiments.generate_commands()
    hela_experiments.to_file("./run_hela.sh")


if __name__ == "__main__":
    main()
