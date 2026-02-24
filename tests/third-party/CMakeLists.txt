# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# MIT License
#
# Modifications Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights
# reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
cmake_minimum_required(VERSION 3.18)
option(CUDA_BACKEND
       "Build for CUDA devices. Default configuration builds for AMD devices."
       OFF)

find_package(ZLIB)
find_path(LIBDEFLATE_INCLUDE_DIR NAMES libdeflate.h)
find_library(LIBDEFLATE_LIBRARY NAMES deflate)

if(CUDA_BACKEND)
  find_package(CUDAToolkit REQUIRED)
  find_package(nvcomp REQUIRED)
  project(nvcomp-tests LANGUAGES CUDA CXX)
  set(HIPCOMP_LIBRARY nvcomp::nvcomp)
else()
  find_package(hipcomp REQUIRED)
  project(nvcomp-tests LANGUAGES HIP CXX)
  set(HIPCOMP_LIBRARY hipcomp)
endif()

# Add Deflate CPU compression GPU decompression test
if(ZLIB_FOUND
   AND LIBDEFLATE_INCLUDE_DIR
   AND LIBDEFLATE_LIBRARY)
  set(TESTNAME deflate_cpu_compression_test)
  add_executable(${TESTNAME} ${TESTNAME}.cu)
  if(CUDA_BACKEND)
    target_compile_definitions(${TESTNAME} PRIVATE CUDA_BACKEND)
    if(NOT MSVC)
      target_link_libraries(
        ${TESTNAME} PRIVATE ${HIPCOMP_LIBRARY} ${LIBDEFLATE_LIBRARY} ZLIB::ZLIB
                            CUDA::cudart_static)
    else()
      # Including CUDA::cudart on Windows resulted in multiple definition linker
      # errors for gdeflate and bitcomp tests, since cudart_static.lib is
      # already included.
      target_link_libraries(
        ${TESTNAME} PRIVATE ${HIPCOMP_LIBRARY} ${LIBDEFLATE_LIBRARY} ZLIB::ZLIB)
    endif()
  else()
    set_source_files_properties(${TESTNAME}.cu PROPERTIES LANGUAGE HIP)
    target_link_libraries(
      ${TESTNAME} PRIVATE ${HIPCOMP_LIBRARY} ${LIBDEFLATE_LIBRARY} ZLIB::ZLIB
                          hip::host hip::device)
  endif()
  target_include_directories(${TESTNAME} PRIVATE ${LIBDEFLATE_INCLUDE_DIR})
  add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
else()
  message(
    WARNING
      "Skipping building Deflate CPU compression GPU decompression test, as the zlib and/or libdeflate libraries were not found."
  )
endif()

# Add GZIP CPU compression GPU decompression test
if(ZLIB_FOUND
   AND LIBDEFLATE_INCLUDE_DIR
   AND LIBDEFLATE_LIBRARY)
  set(TESTNAME gzip_cpu_compression_test)
  add_executable(${TESTNAME} ${TESTNAME}.cu)
  if(CUDA_BACKEND)
    target_compile_definitions(${TESTNAME} PRIVATE CUDA_BACKEND)
    if(NOT MSVC)
      target_link_libraries(
        ${TESTNAME} PRIVATE ${HIPCOMP_LIBRARY} ${LIBDEFLATE_LIBRARY} ZLIB::ZLIB
                            CUDA::cudart_static)
    else()
      # Including CUDA::cudart on Windows resulted in multiple definition linker
      # errors for gdeflate and bitcomp tests, since cudart_static.lib is
      # already included.
      target_link_libraries(
        ${TESTNAME} PRIVATE ${HIPCOMP_LIBRARY} ${LIBDEFLATE_LIBRARY} ZLIB::ZLIB)
    endif()
  else()
    set_source_files_properties(${TESTNAME}.cu PROPERTIES LANGUAGE HIP)
    target_link_libraries(
      ${TESTNAME} PRIVATE ${HIPCOMP_LIBRARY} ${LIBDEFLATE_LIBRARY} ZLIB::ZLIB
                          hip::host hip::device)
  endif()
  target_include_directories(${TESTNAME} PRIVATE ${LIBDEFLATE_INCLUDE_DIR})
  add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
else()
  message(
    WARNING
      "Skipping building Deflate CPU compression GPU decompression test, as the zlib and/or libdeflate libraries were not found."
  )
endif()

# Add ZSTD CPU compression GPU decompression test
find_path(ZSTD_INCLUDE_DIR NAMES zstd.h)
find_library(ZSTD_LIBRARY NAMES zstd)
if(ZSTD_INCLUDE_DIR AND ZSTD_LIBRARY)
  set(ZSTD_TESTS zstd_cpu_compression_test
                 zstd_cpu_compression_cpu_decompression_test)
  foreach(TESTNAME IN LISTS ZSTD_TESTS)
    add_executable(${TESTNAME} ${TESTNAME}.cu)
    target_include_directories(${TESTNAME} PRIVATE ${ZSTD_INCLUDE_DIR})
    if(CUDA_BACKEND)
      target_compile_definitions(${TESTNAME} PRIVATE CUDA_BACKEND)
      if(NOT MSVC)
        target_link_libraries(
          ${TESTNAME} PRIVATE ${HIPCOMP_LIBRARY} ${ZSTD_LIBRARY}
                              CUDA::cudart_static)
      else()
        # Including CUDA::cudart on Windows resulted in multiple definition
        # linker errors for gdeflate and bitcomp tests, since cudart_static.lib
        # is already included.
        target_link_libraries(${TESTNAME} PRIVATE ${HIPCOMP_LIBRARY}
                                                  ${ZSTD_LIBRARY})
      endif()
    else()
      set_source_files_properties(${TESTNAME}.cu PROPERTIES LANGUAGE HIP)
      target_link_libraries(
        ${TESTNAME} PRIVATE ${HIPCOMP_LIBRARY} ${ZSTD_LIBRARY} hip::host
                            hip::device)
    endif()
    # NOTE: We do not register 'zstd_cpu_compression_cpu_decompression_test' as
    # test. We just build it. It is more like a benchmark.
    if(NOT "${TESTNAME}" STREQUAL "zstd_cpu_compression_cpu_decompression_test")
      add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
    endif()
  endforeach()
else()
  message(
    WARNING
      "Skipping building Zstandard CPU compression GPU decompression test, as the Zstandard library was not found."
  )
endif()
