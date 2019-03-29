set(GPU_RESAMPLING_TESTS
  test_resampling
  )

foreach (TEST ${GPU_RESAMPLING_TESTS})
  cuda_add_executable(${TEST}
    ${TEST}.cpp
    )

  target_link_libraries(${TEST}
    ${GPU_RESAMPLING_LIB_TARGET}
    ${Boost_LIBRARIES}
    )

  add_test(
    NAME ${TEST}
    COMMAND ${TEST}
    )
endforeach ()

file(GLOB GPU_RESAMPLING_TEST_IMAGES "${CMAKE_CURRENT_SOURCE_DIR}/data/*.nii.gz")
foreach (IMAGE_FILE ${GPU_RESAMPLING_TEST_IMAGES})
  file(COPY "${IMAGE_FILE}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/data")
endforeach (IMAGE_FILE)
