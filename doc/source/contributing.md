# Contributor guide

## Bug reports and feature requests

Bug reports and feature requests should be submitted by creating an issue on
[CMICLab][cmiclab-niftynet-issue] or [GitHub][github-niftynet-issue].

[cmiclab-niftynet-issue]: https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/issues/new
[github-niftynet-issue]: https://github.com/NifTK/NiftyNet/issues/new


## Merge requests

All merge requests should be submitted via [CMICLab][cmiclab-niftynet-mr]
or GitHub's new pull request.
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


## Writing unit tests

*This section describes steps to create unit tests for NiftyNet.*

#### 1. Which module to test
Go to [Cmiclab pipeline](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/pipelines) page,
click on the latest successful testing pipeline and check the test coverage report at the bottom of the test log, e.g. a coverage report is available at the last part of this [log](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/-/jobs/35553).
The coverage report lists all untested files (with line numbers of specific statements) in the project.

#### 2. File an issue
Create a new issue indicating that you'll be working on the tests of a particular module.

To avoid duplicated effort, please check the [issue list](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/issues) and
make sure nobody is implementing the unit tests for that module at the moment.
Also make sure the issue description is concise and has specific tasks.

#### 3. Create `[name]_test.py`
Clone NiftyNet and create a dedicated branch (from `dev`) for the unit tests.

For Cmiclab users:
```bash
git clone git@cmiclab.cs.ucl.ac.uk:CMIC/NiftyNet.git
git checkout -b unit-test-for-xxx dev
```

For GitHub users, please fork the project to your workspace and
create a branch.


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


#### 6. Send a merge request
After finishing the local tests, git-push the changes to a Cmiclab branch.
This will trigger CI tests, which will run the unit tests on our test server with Ubuntu Linux + Python 2&3).

Please send a merge request with only relevant changes to a particular unit tests.

---

*Thanks for your contributions :)*
