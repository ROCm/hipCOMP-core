# Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# MIT License
#
# Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
cmake_minimum_required(VERSION 3.21) # HIP language support requires 3.21
cmake_policy(VERSION 3.21.3...3.27)
project(hipcomp LANGUAGES C CXX)

option(CUDA_BACKEND "Build for CUDA devices. Default configuration builds for AMD devices." OFF)
option(BUILD_TESTS "Build unit and end-to-end tests." OFF)
option(BUILD_STATIC "Build a static library." OFF)
option(CG_WORKAROUND "Use HIP cooperative groups workaround that is shipped with this project. Has no effect on CUDA builds." OFF)
option(USE_WARPSIZE_32 "Use wave size 32, e.g., for gfx1100 devices. This option is only applicable for the ROCm backend. Has no effect if the CUDA backend is selected." OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_HIP_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)

list( APPEND CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
)

macro(setup_cuda_build)
  enable_language(CUDA)
  
  message(STATUS "Finding CUDAToolkit")
  find_package(CUDAToolkit REQUIRED)
  include_directories("${CUDAToolkit_INCLUDE_DIRS}")

  if(NOT DEFINED CUB_DIR AND DEFINED ENV{CUB_DIR})
    set(CUB_DIR ENV{CUB_DIR})
  endif()
  
  message(STATUS "Finding CUB")
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "11" AND NOT DEFINED CUB_DIR)
    message(FATAL_ERROR "When using a HIP prior to HIP 11, you must specify "
        "the path of cub via the 'CUB_DIR' variable (e.g., "
        "'cmake -DCUB_DIR=...' or 'export CUB_DIR=...'). You can download cub "
        "from 'https://github.com/ROCmSoftwarePlatform/hipCUB'.")
  endif()
  find_path(CUB_PARENT_DIR NAMES "cub/cub.cuh" PATHS "${CUB_DIR}" "${CUDAToolkit_INCLUDE_DIRS}")
  if(NOT CUB_PARENT_DIR)
    message(FATAL_ERROR "Unable to locate 'cub/cub.cuh'. Please specify a path "
        "to cub via 'CUB_DIR'")
  endif()
  
  include_directories(${CUB_PARENT_DIR})
  
  # HIP definitions
  add_compile_definitions(__HIP_PLATFORM_NVCC__ __HIP_PLATFORM_NVIDIA__)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/hipcomp) # contains a hip subfolder

  # Using '-G' can cause the register count to overflow for some kernels
  # and cause launch failures
  # set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG};-G")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g")
endmacro()

macro(setup_hip_amd_build)
  enable_language(HIP)
  message(STATUS "Finding hipCUB")
  find_package(hipcub REQUIRED)
  # for the CXX compiler
  message(STATUS "Finding HIP")
  find_package(hip REQUIRED)
  add_compile_definitions(__HIP_PLATFORM_HCC__ __HIP_PLATFORM_AMD__)
  if (CG_WORKAROUND)
    add_compile_definitions(CG_WORKAROUND)
  endif()
  include_directories(${HIP_INCLUDE_DIRS})
  if(USE_WARPSIZE_32)
    add_compile_definitions("USE_WARPSIZE_32")
  else()
    add_compile_definitions("ENABLE_HIP_OPT_WARPSIZE64")
  endif()
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endmacro()


################################

if (CUDA_BACKEND)
  setup_cuda_build()
else (CUDA_BACKEND)
  setup_hip_amd_build()
endif (CUDA_BACKEND)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fPIC")
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU" OR
    "${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR
    "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")

    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -ggdb")
endif()

# Using '-G' can cause the register count to overflow for some kernels
# and cause launch failures
# set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG};-G")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -ggdb")
set(CMAKE_HIP_FLAGS_DEBUG "${CMAKE_HIP_FLAGS_DEBUG} -g -ggdb")

if(CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(STATUS "Debug build.")
else()
  message(STATUS "Release build.")
  # this is required, as CMake sends CMAKE_CXX_FLAGS_DEBUG to nvcc
  # if no build type is set (but not to g++ for some reason).
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(HIPCOMP_FIND_QUIET "QUIET")
if(DEFINED DEVEL AND NOT DEVEL EQUAL 0)
  unset(HIPCOMP_FIND_QUIET )
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
    # g++ warnings
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weffc++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
    # this doesn't work with old style CMakeLists and hip
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpedantic")

    # g++ warnings
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Werror")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wextra")
    # this doesn't work with old style CMakeLists and hip
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wpedantic")
  endif()

  # only relevant for CUDA lang
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror=all-warnings")
  endif()

  # TODO: add clang warnings
endif()

if (MSVC)
  # Use "/permissive-" in order to be closer to GCC/Clang in what the MSVC compiler accepts.
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W3>" "$<$<COMPILE_LANGUAGE:CXX>:/permissive->")
endif()


if(NOT DEFINED HIPCOMP_EXTS_ROOT AND DEFINED ENV{HIPCOMP_EXTS_ROOT})
  set(HIPCOMP_EXTS_ROOT "$ENV{HIPCOMP_EXTS_ROOT}")
endif()

if( DEFINED HIPCOMP_EXTS_ROOT )
  unset( HIPCOMP_FIND_QUIET )
  set( ENV{ans_ROOT} ${HIPCOMP_EXTS_ROOT} )
  set( ENV{gdeflate_ROOT} ${HIPCOMP_EXTS_ROOT} )
  set( ENV{bitcomp_ROOT} ${HIPCOMP_EXTS_ROOT} )
endif()

find_package(ans ${HIPCOMP_FIND_QUIET})
if (ans_FOUND)
  message (STATUS "Found ans")
  add_definitions(-DENABLE_ANS)
endif()

find_package(bitcomp ${HIPCOMP_FIND_QUIET})
if (bitcomp_FOUND)
  message (STATUS "Found bitcomp")
  add_definitions(-DENABLE_BITCOMP)
endif()

find_package(gdeflate ${HIPCOMP_FIND_QUIET})
if (gdeflate_FOUND)
  message (STATUS "Found gdeflate")
  add_definitions(-DENABLE_GDEFLATE)

  # Find ZLIB for CPU compression
  find_package(ZLIB)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(COPY include/ DESTINATION ${CMAKE_BINARY_DIR}/include)

# determine version
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/include/hipcomp.h" MAIN_HEADER)
string(REGEX MATCH "#define HIPCOMP_MAJOR_VERSION ([0-9]+)" _ "${MAIN_HEADER}")
set(VER_MAJOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define HIPCOMP_MINOR_VERSION ([0-9]+)" _ "${MAIN_HEADER}")
set(VER_MINOR "${CMAKE_MATCH_1}")
string(REGEX MATCH "#define HIPCOMP_PATCH_VERSION ([0-9]+)" _ "${MAIN_HEADER}")
set(VER_PATCH "${CMAKE_MATCH_1}")

set(HIPCOMP_VERSION "${VER_MAJOR}.${VER_MINOR}.${VER_PATCH}")

message(STATUS "Build hipCOMP version ${HIPCOMP_VERSION}")

if (BUILD_TESTS)
  enable_testing()
endif()

if (UNIX)
  include(GNUInstallDirs)
endif()

add_subdirectory(src)

if (BUILD_TESTS)
  add_subdirectory(tests)
endif()

set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/hipcomp)
if(CUDA_BACKEND)
  set(HIPCOMP_PKG_CONFIG_TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipcomp-config.cmake.nvidia.in)
else(CUDA_BACKEND)
  set(HIPCOMP_PKG_CONFIG_TEMPLATE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/hipcomp-config.cmake.amd.in)
endif(CUDA_BACKEND)


#------
# Build directory exporting
include(CMakePackageConfigHelpers)
configure_package_config_file("${HIPCOMP_PKG_CONFIG_TEMPLATE}"
                              "${CMAKE_CURRENT_BINARY_DIR}/hipcomp-config.cmake"
                              INSTALL_DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
export(TARGETS hipcomp
       FILE "${CMAKE_CURRENT_BINARY_DIR}/hipcomp-targets.cmake"
       )


#------
# Install directory setup
configure_package_config_file("${HIPCOMP_PKG_CONFIG_TEMPLATE}"
                              "${CMAKE_CURRENT_BINARY_DIR}/cmake/hipcomp-config.cmake"
                              INSTALL_DESTINATION "${INSTALL_CONFIGDIR}")
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/hipcompConfigVersion.cmake
    VERSION "${HIPCOMP_VERSION}"
    COMPATIBILITY SameMinorVersion)

install(TARGETS hipcomp EXPORT hipcomp-exports DESTINATION lib)
install(EXPORT hipcomp-exports
        FILE          hipcomp-targets.cmake
        NAMESPACE     hipcomp::
        DESTINATION   "${INSTALL_CONFIGDIR}")


install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/hipcomp/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/hipcomp)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/hipcomp.hpp
              ${CMAKE_CURRENT_SOURCE_DIR}/include/hipcomp.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/cmake/hipcomp-config.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/cmake/hipcompConfigVersion.cmake"
        DESTINATION "${INSTALL_CONFIGDIR}")
