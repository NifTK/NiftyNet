if (NOT NumPy_FOUND)
  find_package(PythonInterp REQUIRED)

  execute_process(COMMAND "${PYTHON_EXECUTABLE}" "-c" "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NumPy_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )

  set(NumPy_FOUND 1)
endif (NOT NumPy_FOUND)
