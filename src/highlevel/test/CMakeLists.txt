##
# Copyright 2020 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

# Modifications Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

function(add_unit_test test_file)
  get_filename_component(test_name "${test_file}" NAME_WE)
  add_executable(${test_name} ${test_file})
  target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR})
  if (CUDA_BACKEND)
    target_link_libraries(${test_name} PRIVATE hipcomp CUDA::cudart)
  else (CUDA_BACKEND)
    target_link_libraries(${test_name} PRIVATE hipcomp hip::host)
  endif (CUDA_BACKEND)
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

file(GLOB tests *_test.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

foreach(file ${tests})
  add_unit_test(${file})
endforeach()