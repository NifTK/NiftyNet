import tensorflow as tf

# This UGLY solution is done to bypass the issue
# outlined in the NiftyNet issue #381 and Tensorflow
# issue #29439
#
# https://github.com/NifTK/NiftyNet/issues/381
# https://github.com/tensorflow/tensorflow/issues/29439

try:
    delattr(tf.test.TestCase,'test_session')
except AttributeError:
    pass


class NiftyNetTestCase(tf.test.TestCase):
    pass