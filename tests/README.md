## Writing unit tests

*This readme file describes steps to create unit tests for NiftyNet.*

#### 1. Find out which NiftyNet modules require unit tests
Go to [Cmiclab pipeline](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/pipelines) page,
click on the latest successful testing pipeline and check the test coverage report at the bottom of the test log, e.g. a coverage report is available at the last part of this [log](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/-/jobs/35553).
The coverage report lists all untested files (with line numbers of specific statements) in the project.

#### 2. File an issue on the issue list
Create a new issue indicating that you'll be working on the tests of a particular module.

To avoid duplicated effort, please check the [issue list](https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/issues) and
make sure nobody is implementing the unit tests for that module at the moment.
Also make sure the issue description is concise and has specific tasks.

#### 3. Create a `[name]_test.py` file in [`NiftyNet/tests/`](../tests) folder
Clone NiftyNet and create a dedicated branch (from `dev`) for the unit tests.
```bash
git clone git@cmiclab.cs.ucl.ac.uk:CMIC/NiftyNet.git
git checkout -b unit-test-for-xxx dev
```
Create a unit test Python script with file name ends with `_test.py` in [`NiftyNet/tests/`](../tests) folder.
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


#### 4. Run the unit test locally
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

#### 5. Run all unit tests locally
Normally the newly created unit test should not depend on the outcome of the other unit tests.
A Bash script is defined [here](../run_test.sh) for running all quick tests to confirm this.

(In `run_test.sh`, `wget` and `tar` are used to automatically download and unzip testing data. This can be done manually.)


#### 6. Push to Cmiclab and send a merge request
After finishing the local tests, git-push the changes to a Cmiclab branch.
This will trigger CI tests, which will run the unit tests on our test server with Ubuntu Linux + Python 2&3).

Please send a merge request with only relevant changes to a particular unit tests.

---

*Thanks for your contributions :)*
