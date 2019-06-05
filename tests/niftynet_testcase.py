import tensorflow as tf

try:
    delattr(tf.test.TestCase,'test_session')
except AttributeError:
    pass


class NiftyNetTestCase(tf.test.TestCase):
    pass