## NiftyNet

The main source code repository for NiftyNet is [CMICLab][cmiclab-niftynet].
The NiftyNet codebase is also mirrored on [GitHub][github-niftynet].

[cmiclab-niftynet]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet
[github-niftynet]: https://github.com/NifTK/NiftyNet


## Submitting bug reports and feature requests

Bug reports and feature requests should be submitted by creating an issue on [CMICLab][cmiclab-niftynet-issue] or [GitHub][github-niftynet-issue].

[cmiclab-niftynet-issue]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/issues/new
[github-niftynet-issue]: https://github.com/NifTK/NiftyNet/issues/new


## Submitting merge requests

All merge requests should be submitted via [CMICLab][cmiclab-niftynet-mr] or
GitHub pull request.
Please make sure you have read the following subsections before submitting a merge request.

[cmiclab-niftynet-mr]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/merge_requests/new


### Python style guide

Please follow the [PEP8 Style Guide for Python Code][pep8].
In particular (from the guide):

> Please be consistent.
> If you're editing code, take a few minutes to look at the code around you and
> determine its style. If they use spaces around all their arithmetic operators,
> you should too. If their comments have little boxes of hash marks around them,
> make your comments have little boxes of hash marks around them too.

[pep8]: https://www.python.org/dev/peps/pep-0008/


### Testing your changes

Please submit merge requests from your branch to the `dev` branch.

Before submitting a merge request, please make sure your branch passes all
unit tests, by running:

``` sh
cd NiftyNet/
sh run_test.sh
```

**If you have made changes to the pip installer, please test these.**
For instance if you have added a new [CLI entry point][pip-console-entry]  (i.e. a new "command" - also see the respective section below), make sure you include the appropriate tests in the [GitLab CI configuration][gitlab-ci-yaml].
For an example how to do this please see [lines 223 to 270 in the `.gitlab-ci.yml` file][gitlab-ci-pip-installer-test].

[pip-console-entry]: http://python-packaging.readthedocs.io/en/latest/command-line-scripts.html#the-console-scripts-entry-point
[gitlab-ci-yaml]: https://docs.gitlab.com/ce/ci/yaml/
[gitlab-ci-pip-installer-test]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/940d7a827d6835a4ce10637014c0c36b3c980476/.gitlab-ci.yml#L223


## Enhancing the pip installer

### Adding a new command callable from a pip-installed NiftyNet

This requires added a new [`console_scripts` entry point][pip-console-entry] in the `setup.py` file.
For a practical example see [how the `net_segment` CLI command is implemented][net-segment-entry].
Also see [how this command is tested][net-segment-test].

[net-segment-entry]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/940d7a827d6835a4ce10637014c0c36b3c980476/setup.py#L105
[net-segment-test]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/940d7a827d6835a4ce10637014c0c36b3c980476/.gitlab-ci.yml#L252


## Writing unit tests
Please see this [README](tests/README.md) for more information on how to write unit tests for NiftyNet.

## NiftyNet admin tasks

### Making a release

NiftyNet versions are numbered following [Semantic Versioning (semver)](http://semver.org/spec/v2.0.0.html).
After adding notes for the current release to the [NiftyNet changelog][changelog], the current release should be [tagged][git-tag] with a [PEP440][pep440]-compliant semver number preceded by the letter `v` (for "version").

[pep440]: https://www.python.org/dev/peps/pep-0440/
[changelog]: CHANGELOG.md

### Bundling a pip installer

The NiftyNet pip installer gets bundled automatically for [Git tags][git-tag] starting with a `v` (for "version") pushed to [CMICLab][niftynet-cmiclab].
The [wheel version][wheel-version-tag] is determined automatically as part of this process.
To see how this is done in practice, please go to the [`pip-camera-ready` section of `.gitlab-ci.yml`][pip-camera-ready] (and see the result in [this build log - esp. the last few lines lines, which show where the pip installer can be found on the build server][pip-camera-ready-output]).

In particular, bundling a pip installer boils down to running the command [`python setup.py bdist_wheel`][python-setuptools] in the top-level directory.
This creates a [wheel binary package][wheel-binary] in a newly created `dist` directory, e.g. `dist/NiftyNet-0.2.0-py2.py3-none-any.whl`.

[niftynet-cmiclab]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet
[git-tag]: https://git-scm.com/book/en/v2/Git-Basics-Tagging
[pip-camera-ready]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/940d7a827d6835a4ce10637014c0c36b3c980476/.gitlab-ci.yml#L323
[pip-camera-ready-output]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/-/jobs/30450
[python-setuptools]: https://packaging.python.org/tutorials/distributing-packages/#wheels
[wheel-binary]: https://www.python.org/dev/peps/pep-0491/


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

The main development hub for NiftyNet is [CMICLab][cmiclab-niftynet].
However we would also like to support the [GitHub][github-niftynet]-based workflow in a way that is minimally disruptive to the workflow on CMICLab.
For this purpose, please follow the steps below for merging pull requests on GitHub:

1. **[on GitHub]** Review the pull request, and ask for changes if needed
1. **[on GitHub]** Accept the pull request (i.e. click the "Merge pull request" button)
1. **[on CMICLab]** Create a new branch off `dev` with a name representative of the pull request. For instance `merging-github-pr-7` if the pull request on GitHub was numbered `7` (assuming `origin` is set to `git@cmiclab.cs.ucl.ac.uk:CMIC/NiftyNet.git`):
   * `git checkout -b merging-github-pr-7 origin/dev`
1. **[on CMICLab]** Pull GitHub's `dev` branch onto the new branch `merging-github-pr-7` you've created (assuming `origin` is set to `git@cmiclab.cs.ucl.ac.uk:CMIC/NiftyNet.git`) **and** push this new branch to CMICLab:
   1. `git pull git@github.com:NifTK/NiftyNet.git dev`
   1. `git push -u origin merging-github-pr-7`
1. **[on CMICLab]** Make sure `merging-github-pr-7` passes all [continuous integration tests on CMICLab][cmiclab-niftynet-pipelines]
1. **[on CMICLab]** Merge the new branch `merging-github-pr-7` onto `dev`
1. **[on GitHub]** Check that the last step has updated the `dev` branch mirror

[cmiclab-niftynet-pipelines]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/pipelines
