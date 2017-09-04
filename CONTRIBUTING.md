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
(The `run_test.sh` script requires `wget` and `tar` to prepare testing images.)
