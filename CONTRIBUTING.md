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

All merge requests should be submitted via [CMICLab][cmiclab-niftynet-mr].
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
For instance if you have added a new [CLI entry point (i.e. a new "command")][pip-console-entry], make sure you include the appropriate tests in the [GitLab CI configuration][gitlab-ci-yaml].
For an example how to do this please see [lines 223 to 270 in the `.gitlab-ci.yml` file][gitlab-ci-pip-installer-test].

[pip-console-entry]: http://python-packaging.readthedocs.io/en/latest/command-line-scripts.html#the-console-scripts-entry-point
[gitlab-ci-yaml]: https://docs.gitlab.com/ce/ci/yaml/
[gitlab-ci-pip-installer-test]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/940d7a827d6835a4ce10637014c0c36b3c980476/.gitlab-ci.yml#L223

### Publishing a NiftyNet pip installer on PyPI

