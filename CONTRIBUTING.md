# Contributor guide
The source code for NiftyNet is released via [GitHub][github-niftynet].

[github-niftynet]: https://github.com/NifTK/NiftyNet


- [Submitting bug reports and feature requests](#submitting-bug-reports-and-feature-requests)
- [Submitting merge requests](#submitting-merge-requests)
    - Python style guide
    - Test your changes
    - Create GitHub pull requests
- [Submitting model zoo entries](#submitting-model-zoo-entries)
    - Fork the model zoo repo
    - Create a new folder and add model zoo data
    - Update documentation
    - Create GitHub pull requests
- [Writing unit tests](#writing-unit-tests)
    - Determine which module to test
    - File an issue
    - Create `[name]_test.py`
    - Run tests locally
    - Run all tests locally
- [NiftyNet admin tasks](#niftynet-admin-tasks)
    - Making a release
    - Publishing a NiftyNet pip installer on PyPI
    - Merging GitHub pull requests
    - Enhancing the pip installer
    - Bundling a pip installer

## Submitting bug reports and feature requests

Bug reports and feature requests should be submitted by creating an issue on [GitHub][github-niftynet-issue].

[github-niftynet-issue]: https://github.com/NifTK/NiftyNet/issues/new?template=niftynet-issue-template.md


## Submitting merge requests

All merge requests should be submitted via GitHub pull request.

Please make sure you have read the following subsections before submitting a merge request.


### 1. Python style guide

Please follow the [PEP8 Style Guide for Python Code][pep8].
In particular (from the guide):

> Please be consistent.
> If you're editing code, take a few minutes to look at the code around you and
> determine its style. If they use spaces around all their arithmetic operators,
> you should too. If their comments have little boxes of hash marks around them,
> make your comments have little boxes of hash marks around them too.

[pep8]: https://www.python.org/dev/peps/pep-0008/


### 2. Test your changes

Please submit merge requests from your branch to the `dev` branch.

Before submitting a merge request, please make sure your branch passes all
unit tests, by running:

``` sh
cd NiftyNet/
sh run_test.sh
```

### 3. Create GitHub pull requests
1. **[on GitHub]** Sign up/in [GitHub.com](https://github.com/) (The rest steps assume GitHub user id: `nntestuser`).
1. **[on GitHub]** Go to [https://github.com/NifTK/NiftyNet](https://github.com/NifTK/NiftyNet), click the 'Fork' button.
1. Download the repo:
   * `git clone https://github.com/nntestuser/NiftyNet.git`
1. Synchronise your repo with the `dev` branch of [https://github.com/NifTK/NiftyNet](https://github.com/NifTK/NiftyNet):
   * `git remote add upstream git@github.com:NifTK/NiftyNet.git`
   * `git pull upstream dev`
1. Make commits, test changes locally, and push to `nntestuser`'s repo:
   * `git push github dev`

   (This step assumes `github` is a remote name pointing at `git@github.com:nntestuser/NiftyNet.git`;

    set this with command: `git remote add github git@github.com:nntestuser/NiftyNet.git`;

    confirm this with command: `git remote -v`)

1. **[on GitHub]** Create a pull request by clicking the 'pull request' button.


## Submitting model zoo entries
NiftyNet provides a version-controlled model zoo deployed on GitHub,
we welcome new model zoo entry submissions!

The model zoo is itself a GitHub project, the workflow of submitting new entries is in general the same as sending a
[GitHub pull request](https://help.github.com/articles/fork-a-repo/).

The following is a step-by-step guide for submitting a new entry named `foo_bar_model_zoo`. 
After finishing these steps, all users will be able to download the model by running
NiftyNet command `net_download foo_bar_model_zoo`. 

*`foo_bar_model_zoo` is a model zoo entry ID for demo purposes only, normally we prefer meaningful IDs, 
which should briefly indicate the method, network architecture, and the task name.*

#### 1.  Fork the model zoo repo
NiftyNet model zoo uses [Git Large File Storage -- git-lfs](https://git-lfs.github.com/) for large file (such as trained network weights) versioning.
Make sure you have installed `git-lfs` and file archiving tool `tar` beforehand.

[Fork and `git clone` the repo](https://help.github.com/articles/fork-a-repo/) 
to your local machine, create a new folder called `foo_bar` within the codebase.
This folder will hold all the new content of the proposed entry.

#### 2. Create a new folder and add model zoo data
The new `foo_bar_model_zoo` entry can be NiftyNet application configuration files, demo image data, or some trained weights, 
or a combination of these data.  They should be archived into at most three `.tar.gz` files:
- `data.tar.gz`: for training/inference images, this will eventually go to the user's `~/niftynet/data/foo_bar` folder by default.
- `config.tar.gz`: for customised Python code, such as new loss functions, image samplers, as well as application configuration file.
This will go to the users `~/niftynet/extensions/foo_bar` folder by default.
- `weights.tar.gz`: for the trained weights, this will go to the user's `~/niftynet/models/foo_bar` folder by default.

We recommend that the `.tar.gz` files to be created by running, for example
```bash
tar -cvzf ../data.tar.gz ./input_demo_data*.nii
```
The command will create an archive outside the current directory which contains the image `input_demo_data.nii` .
Un-archiving this file will output the images with filename matched the patten `input_demo_data*.nii`.

Similarly this can be done for the configuration files as well as Python code:
```bash
tar -cvzf ../config.tar.gz ./myconfig*.ini
```

For the trained weights, we require the following specific folder structure:
```
└── foo_bar
    ├── databrain_std_hist_models_otsu.txt
    └── models
        ├── model.ckpt-33000.data-00000-of-00001
        ├── model.ckpt-33000.index
        └── model.ckpt-33000.meta
```
Where `databrain_std_hist_models_otsu.txt` is a label or intensity histogram mapping file generated by NiftyNet (if applicable);
`models` and `model.ckpt-*` names are compulsory: NiftyNet will always look for the `models` folder when reading the model zoo entry. 

After having this folder structure, the archive file can be created by running:
```bash
cd your_trained_model_folder/
tar -cvzf ../weights.tar.gz ./*
```

The outcome of this step should be several `.tar.gz` files within the `foo_bar` folder, within the cloned model zoo GitHub project:
```
└── foo_bar
    ├── data.tar.gz
    ├── config.tar.gz
    └── weights.tar.gz
```

#### 2.5 Make a `main.ini`
Within `foo_bar` folder, create a `main.ini` file, with optional sections of
`[code]`, `[data]`, and `[weights]`. So that the end-users' `net_download` command 
knows where to fetch and un-archive the shared data.

Each section should have the following values
```ini
[code]
# should be the model zoo entry name
local_id = foo_bar
# the actual url for the .tar.gz
url =  https://github.com/NifTK/NiftyNetModelZoo/raw/5-reorganising-with-lfs/foo_bar/config.tar.gz
# `action` is a reserved keyword, only `expanding` action is currently supported
action = expand
# available options are [models|extensions|data]
destination = models
```
This config section will be effectively parsed by `net_download` as:
1. download data from `https://github.com/NifTK/NiftyNetModelZoo/raw/5-reorganising-with-lfs/foo_bar/config.tar.gz` (`url`),
2. un-archiving the downloaded data (`action`), 
3. create a new folder in `~/niftynet/models` named `foo_bar` (`destination` and `local_id`).
4. copy the downloaded data to `~/niftynet/models/foo_bar`.

#### 3. Update documentation
Make a readme file named `README.md` in the `foo_bar` folder, make sure that you included
appropriate references, licenses information about the data you're sharing.

#### 4. Create GitHub pull requests
As a result of the previous steps, you should have created a new `foo_bar` entry with the following folder structure:
```
└── foo_bar
    ├── main.ini
    ├── README.md
    ├── data.tar.gz
    ├── config.tar.gz
    └── weights.tar.gz
```
Now you can [send a pull request](https://help.github.com/articles/creating-a-pull-request/) to https://github.com/NifTK/NiftyNetModelZoo.


## Writing unit tests
*This section describes steps to create unit tests for NiftyNet.*

#### 1. Determine which module to test
Go to [Gitlab pipeline](https://gitlab.com/NifTK/NiftyNet/pipelines) page,
click on the latest successful testing pipeline and check the test coverage report at the bottom of the test log.
The coverage report lists all untested files (with line numbers of specific statements) in the project.


#### 2. File an issue
Create a new issue indicating that you'll be working on the tests of a particular module.

To avoid duplicated effort, please check the [issue list](https://github.com/NifTK/NiftyNet/issues) and
make sure nobody is implementing the unit tests for that module at the moment.
Also make sure the issue description is concise and has specific tasks.

#### 3. Create `[name]_test.py`
Create a unit test Python script with file name ends with `_test.py`. This file
should be added to
[`NiftyNet/tests/`](https://github.com/NifTK/NiftyNet/tree/dev/tests) directory.
(CI runner will automatically pick up the script and run it with Python 2.7&3)

A minimal working template for  `[name]_test.py` is:
```python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import tensorflow as tf

class ModuleNameTest(tf.test.TestCase):
    def test_my_function(self):
        x = tf.constant(1.0)
        self.assertEqual(x.eval(), 1.0)
    # preferably using self.assert* functions from TensorFlow unit tests API
    # https://www.tensorflow.org/versions/r0.12/api_docs/python/test/unit_tests

if __name__ == "__main__":
    # so that we can run this test independently
    tf.test.main()
```
If the unit tests write files locally, please ensure it's writing to `NiftyNet/testing_data` folder.


#### 4. Run tests locally
In NiftyNet source code folder, run:
```bash
python -m tests.[name]_test.py
```
make sure the test works locally.
The test should finish in a few seconds (using CPU). If it takes significantly longer, please set it as `slow test` in the file:
```python
...
@unittest.skipIf(os.environ.get('QUICKTEST', "").lower() == "true", 'Skipping slow tests')
class ModuleNameTest(tf.test.TestCase):
    def test_my_function(self):
        pass
    # preferably using self.assert* functions from tensorflow unit tests API
    # https://www.tensorflow.org/versions/r0.12/api_docs/python/test/unit_tests
...
```

#### 5. Run all tests locally
Normally the newly created unit test should not depend on the outcome of the other unit tests.
[A Bash script is defined](https://github.com/NifTK/NiftyNet/blob/dev/run_test.sh) for running all quick tests to confirm this.

(In `run_test.sh`, `wget` and `tar` are used to automatically download and unzip testing data. This can be done manually.)

Please send a merge request with only relevant changes to a particular unit tests.

---

*Thanks for your contributions :)*


## NiftyNet admin tasks

### Making a release

NiftyNet versions are numbered following [Semantic Versioning (semver)](http://semver.org/spec/v2.0.0.html).
After adding notes for the current release to the [NiftyNet changelog][changelog], the current release should be [tagged][git-tag] with a [PEP440][pep440]-compliant semver number preceded by the letter `v` (for "version").

Steps to release a new version:
1. Prepare and proofread a draft release note;
1. Add release note to the changelog in the [Changelog][keepachangelog] format;
    * Update the `[Unreleased]` link in the changelog,
    * Append a GitHub comparison URL entry to the changelog file;
1. Push the release note changes to a new branch `releasing-x`;
1. Send a pull request from `releasing-x` to `dev`;
1. Check CI tests outcome, check changelog, accept the pull request;
1. Tag the latest commit of `dev` (make sure that commit is not [skip][gitlab-ci-skip]ped, as this will subsequently [skip the tag build][tag-ci-skip-issue]);
1. Once the tag has been pushed to GitHub, run [chandler][chandler] to synchronise the changelog with the published release on GitHub
1. the `pip stage` will be triggered in CI, there should be a wheel ready;
1. Publish the pip wheel on [PyPI test server][pypi-test];
1. Inspect testing front page, make sure everything looks fine, links work, etc.;
1. Push pip wheel to release (warning: not revertible);
1. Merge `dev` to `master` (archiving the new version).

[tag-ci-skip-issue]: https://gitlab.com/gitlab-org/gitlab/issues/18798
[gitlab-ci-skip]: https://docs.gitlab.com/ee/ci/yaml/README.html#skipping-jobs
[chandler]: https://github.com/mattbrictson/chandler
[pep440]: https://www.python.org/dev/peps/pep-0440/
[changelog]: CHANGELOG.md
[git-tag]: https://git-scm.com/book/en/v2/Git-Basics-Tagging
[keepachangelog]: http://keepachangelog.com/en/1.0.0/
[pypi-test]: https://test.pypi.org/

### Publishing a NiftyNet pip installer on PyPI

Making NiftyNet available to the world via a simple `pip install niftynet` requires publishing the created wheel on the [Python Package Index (PyPI)][niftynet-pypi].
**BUT PLEASE TAKE YOUR TIME TO READ THE NOTES BELOW BEFORE PROCEEDING:**

* PyPI is very tightly coupled to [package versions][wheel-version-tag].
That means, once a wheel tagged e.g. as version `1.0.1` has been published, it is final.
In other words, **you cannot change your source code, bundle it again using the same version and re-submit to PyPI as the "updated" version `1.0.1`**.
* Please consider submitting the bundled wheel to the [PyPI test site][uploading-to-pypi] (see the [NiftyNet test page][niftynet-pypi-test]) to assess the visual appearance of the PyPI page before publishing on the actual PyPI.

[wheel-version-tag]: https://www.python.org/dev/peps/pep-0491/#file-name-convention
[niftynet-pypi]: https://pypi.org/project/NiftyNet/
[niftynet-pypi-test]: https://test.pypi.org/project/NiftyNet/
[uploading-to-pypi]: https://packaging.python.org/tutorials/distributing-packages/#uploading-your-project-to-pypi

To actually publish the bundled wheel on PyPI, you will need to run the `twine upload` command e.g. `twine upload dist/NiftyNet-0.2.0-py2.py3-none-any.whl` - this will of course work only if you have set the corresponding [PyPI account credentials][pypi-create-account].

[pypi-create-account]: https://packaging.python.org/tutorials/distributing-packages/#create-an-account


### Merging GitHub pull requests

Please follow the steps below for merging pull requests on GitHub:

1. **[on GitHub]** Review the pull request, and ask for changes if needed.
1. Create a new branch off `dev` of `https://github.com/NifTK/NiftyNet` with a name representative of the pull request.
   For instance, if the pull request on GitHub was numbered `7` (assuming `upstream` is set to `git@github.com:NifTK/NiftyNet.git`):
   * `git checkout -b merging-github-pr-7 upstream/dev`
1. Download the contributing commits and merge to `merging-pr-7`.
   For instance, if the pull request is from `nntestuser`'s `bug-fixing-branch`:
   * `git pull https://github.com/nntestuser/NiftyNet bug-fixing-branch`
1. Review and test locally.
1. Push the commits to branch `merging-github-pr-7` of remote repository [https://github.com/NifTK/NiftyNet](https://github.com/NifTK/NiftyNet):
   * `git push upstream merging-github-pr-7`

1. **[on GitHub]** Check CI tests results ([Gitlab.com](https://gitlab.com/NifTK/NiftyNet/pipelines); quick tests only).
1. **[on GitHub]** Create a new pull request from `merging-github-pr-7` to `dev`.
1. **[on GitHub]** Accept the new pull request onto `dev`.
1. **[on GitHub]** Check CI tests results ([Gitlab.com](https://gitlab.com/NifTK/NiftyNet/pipelines); full tests for `dev`)

*At the moment only pushes (instead of pull requests from forks) to GitHub
trigger GitLab's CI runner, [a feature
request](https://gitlab.com/gitlab-org/gitlab-ee/issues/6775) has been
submitted -- will simplify the workflow once resolved ([more
info](https://github.com/NifTK/NiftyNet/issues/120#issuecomment-401531891)).*

## Enhancing the pip installer

### Adding a new command callable from a pip-installed NiftyNet

This requires added a new [`console_scripts` entry point][pip-console-entry] in the `setup.py` file.
For a practical example see [how the `net_segment` CLI command is implemented][net-segment-entry].

[pip-console-entry]: http://python-packaging.readthedocs.io/en/latest/command-line-scripts.html#the-console-scripts-entry-point
[net-segment-entry]: https://github.com/NifTK/NiftyNet/blob/v0.3.0/setup.py#L107

