if (NOT Tensorflow_FOUND)
  find_package(PythonInterp REQUIRED)

  macro (get_tensorflow_setting OUTVAR SETTING_GETTER)
    execute_process(
      COMMAND "${PYTHON_EXECUTABLE}" -c "import tensorflow; ${SETTING_GETTER}"
      OUTPUT_VARIABLE ${OUTVAR}
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
      TIMEOUT 120
      )
  endmacro (get_tensorflow_setting)

  get_tensorflow_setting(Tensorflow_INCLUDE_DIRS "print(tensorflow.sysconfig.get_include())")
  get_tensorflow_setting(Tensorflow_LIBRARY_DIRS "print(tensorflow.sysconfig.get_lib())")
  get_tensorflow_setting(Tensorflow_CFLAGS "print(' '.join(tensorflow.sysconfig.get_compile_flags()))")
  get_tensorflow_setting(Tensorflow_LFLAGS "print(' '.join(tensorflow.sysconfig.get_compile_link()))")

  set(Tensorflow_LIBRARIES
    "tensorflow_framework")

  if (Tensorflow_INCLUDE_DIRS)
    set(Tensorflow_FOUND 1)
  endif (Tensorflow_INCLUDE_DIRS)
endif (NOT Tensorflow_FOUND)
