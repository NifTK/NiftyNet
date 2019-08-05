import os.path as osp
import tensorflow as tf

__MODULE_FILENAME__ = '$<TARGET_FILE_NAME:niftyreg_image_resampling_ops>'
__INSTALL_LOCATION__ \
    = '$<TARGET_PROPERTY:niftyreg_image_resampling_ops,INSTALL_NAME_DIR>'
__BUILD_LOCATION__ \
    = '$<TARGET_PROPERTY:niftyreg_image_resampling_ops,BINARY_DIR>'


def _find_and_load_op():
    for test_path in (osp.join(osp.dirname(__file__), __MODULE_FILENAME__),
                      osp.join('.', __INSTALL_LOCATION__, __MODULE_FILENAME__),
                      osp.join(__INSTALL_LOCATION__, __MODULE_FILENAME__),
                      osp.join(__BUILD_LOCATION__, __MODULE_FILENAME__)):
        if osp.isfile(test_path):
            return tf.load_op_library(test_path)

    # raise ImportError('Could not locate resampling library: %s'
    #                   % __MODULE_FILENAME__)


__nr_module__ = _find_and_load_op()


def get_niftyreg_module():
    return __nr_module__
