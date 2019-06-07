project(NiftyNet_gpu_resampling)
cmake_minimum_required(VERSION 3.5)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules;${CMAKE_MODULE_PATH}")

set(GPU_RESAMPLING_CONFIGFILE_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE STRING "Destination directory for configured files.")

add_definitions("-DGOOGLE_CUDA")
set(GPU_RESAMPLING_LIB_TARGET niftyreg_image_resampling_ops)
set(CMAKE_CXX_STANDARD 11)
# This may not be portable (almost certainly isn't)!
# -DNDEBUG: a constexpr applied to function returning std::string is causing the build to fail; the code appears to be inside an assert, so disabling asserts "solves" the problem.
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11 --expt-relaxed-constexpr -DNDEBUG --disable-warnings")

find_package(CUDA)
find_package(Tensorflow REQUIRED)

include_directories(nifti)
include_directories(SYSTEM "${Tensorflow_INCLUDE_DIRS}")
link_directories("${Tensorflow_LIBRARY_DIRS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${Tensorflow_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${Tensorflow_CFLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${Tensorflow_CFLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}")

set(GPU_RESAMPLING_CXX_SRC
  _reg_resampling.cpp
  _reg_tools.cpp
  _reg_maths.cpp

  niftyreg_cpu_resample_op.cpp
  niftyreg_cpu_resample_gradient_op.cpp
  niftyreg_cpu_resample_image_gradient_op.cpp

  nifti/nifti1_io.c
  nifti/znzlib.c
  )

if (CUDA_FOUND)
  set(GPU_RESAMPLING_CU_SRC
    resampleKernel.cu
    _reg_common_cuda.cu
    _reg_resampling_gpu.cu

    niftyreg_gpu_resample_op.cu
    niftyreg_gpu_resample_gradient_op.cu
    )

  cuda_add_library(${GPU_RESAMPLING_LIB_TARGET} SHARED
    ${GPU_RESAMPLING_CXX_SRC}
    ${GPU_RESAMPLING_CU_SRC}
    )
else ()
  message(WARNING "No CUDA toolkit was found, if your tensorflow install was built with CUDA support, the build will fail.")
  add_library(${GPU_RESAMPLING_LIB_TARGET} SHARED
    ${GPU_RESAMPLING_CXX_SRC}
    )
endif (CUDA_FOUND)

set_target_properties(${GPU_RESAMPLING_LIB_TARGET} PROPERTIES PREFIX "")
target_link_libraries(${GPU_RESAMPLING_LIB_TARGET}
  ${Tensorflow_LIBRARIES}
  )
file(GENERATE OUTPUT "${GPU_RESAMPLING_CONFIGFILE_DIR}/niftyreg_module_loader.py" INPUT "${CMAKE_CURRENT_SOURCE_DIR}/../niftyreg_module_loader.py.in")
