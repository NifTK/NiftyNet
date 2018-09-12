# Contributor guide
The source code for NiftyNet is released via [GitHub][github-niftynet].

[github-niftynet]: https://github.com/NifTK/NiftyNet


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


### 2. Testing your changes

Please submit merge requests from your branch to the `dev` branch.

Before submitting a merge request, please make sure your branch passes all
unit tests, by running:

``` sh
cd NiftyNet/
sh run_test.sh
```

### 3. Creating GitHub pull requests
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
python -m tests.[name]_test
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

NiftyNet versions are numbered following [Semantic Versioning (semver)](https://semver.org/spec/v2.0.0.html).
After adding notes for the current release to the [NiftyNet changelog][changelog], the current release should be [tagged][git-tag] with a [PEP440][pep440]-compliant semver number preceded by the letter `v` (for "version").

[pep440]: https://www.python.org/dev/peps/pep-0440/
[changelog]: CHANGELOG.md

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

[net-segment-entry]: https://github.com/NifTK/NiftyNet/blob/v0.3.0/setup.py#L107


## Bundling a pip installer

The NiftyNet pip installer gets bundled automatically for [Git tags][git-tag] starting with a `v` (for "version").
The [wheel version][wheel-version-tag] is determined automatically as part of this process.
The last few lines of the CI build log show the location of the bundled pip installer on the server, e.g.:

```bash
$ echo "Camera-ready pip installer bundle (wheel) created:"
Camera-ready pip installer bundle (wheel) created:
$ echo "$(ls $camera_ready_dir/*.whl)"
/home/gitlab-runner/environments/niftynet/pip/camera-ready/NiftyNet-0.2.0-py2.py3-none-any.whl
Job succeeded
```

In particular, bundling a pip installer boils down to running the command [`python setup.py bdist_wheel`][python-setuptools] in the top-level directory.
This creates a [wheel binary package][wheel-binary] in a newly created `dist` directory, e.g. `dist/NiftyNet-0.2.0-py2.py3-none-any.whl`.

[git-tag]: https://git-scm.com/book/en/v2/Git-Basics-Tagging
[python-setuptools]: https://packaging.python.org/tutorials/distributing-packages/#wheels
[wheel-binary]: https://www.python.org/dev/peps/pep-0491/


**If you have made changes to the pip installer, please test these.**
For instance if you have added a new [CLI entry point][pip-console-entry]  (i.e. a new "command" - also see the respective section below),
make sure you include the appropriate tests in the [GitLab CI configuration][gitlab-ci-yaml].
For an example how to do this please see [lines 223 to 270 in the `.gitlab-ci.yml` file][gitlab-ci-pip-installer-test].

[pip-console-entry]: http://python-packaging.readthedocs.io/en/latest/command-line-scripts.html#the-console-scripts-entry-point
[gitlab-ci-yaml]: https://docs.gitlab.com/ce/ci/yaml/
[gitlab-ci-pip-installer-test]: https://github.com/niftk/NiftyNet/blob/940d7a827d6835a4ce10637014c0c36b3c980476/.gitlab-ci.yml#L223
