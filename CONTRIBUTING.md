The main source code repository for NiftyNet is [CMICLab][cmiclab-niftynet].
All merge requests should be submitted via CMICLab.

[cmiclab-niftynet]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet

We'd like to follow this python style guide: [pep8]

> Please be consistent.
> If you're editing code, take a few minutes to look at the code around you and
> determine its style. If they use spaces around all their arithmetic operators,
> you should too. If their comments have little boxes of hash marks around them,
> make your comments have little boxes of hash marks around them too.

[pep8]: https://www.python.org/dev/peps/pep-0008/

When submitting merge requests, please merge to the dev branch.

Before submitting a merge request, please make sure your branch passes all
unit tests, by running:
``` sh
cd NiftyNet/
sh run_test.sh
```
(`run_test.sh` script requires `wget` and `tar` to prepare testing images.)
