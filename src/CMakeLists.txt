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
#

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

macro (setup_cuda_devices)
  # build real and virtual, so as to run on 6x cards.
  set(GPU_ARCHS "60")

  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "8")
    set(GPU_ARCHS ${GPU_ARCHS} "70-real")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER "9")
    set(GPU_ARCHS ${GPU_ARCHS} "75-real")
  endif()
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "11")
    set(GPU_ARCHS ${GPU_ARCHS} "80")
  endif()
endmacro(setup_cuda_devices)

macro (set_cuda_compilation_options)
  set_property(TARGET hipcomp PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_property(TARGET hipcomp PROPERTY CUDA_ARCHITECTURES ${GPU_ARCHS})
  target_compile_options(hipcomp PRIVATE
      $<$<COMPILE_LANGUAGE:HIP>:--expt-extended-lambda>)
  if(DEFINED CUB_DIR)
    target_include_directories(hipcomp PRIVATE "${CUB_DIR}")
  endif()
  target_include_directories(hipcomp PRIVATE "${CUDAToolkit_INCLUDE_DIRS}")
endmacro (set_cuda_compilation_options)

file(GLOB HIP_SOURCES
  ./*.hip
  ./lowlevel/*.hip
  ./highlevel/*.hip)
file(GLOB CPP_SOURCES
  ./*.cpp
  ./lowlevel/*.cpp
  ./highlevel/*.cpp)

if (CUDA_BACKEND)
  setup_cuda_devices()
  set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE CUDA)
endif (CUDA_BACKEND)

if (BUILD_STATIC)
  add_library(hipcomp STATIC ${HIP_SOURCES} ${CPP_SOURCES})
else()
  if (MSVC_VERSION)
    # we have to export all symbols on windows for a dll
    add_library(hipcomp STATIC ${HIP_SOURCES} ${CPP_SOURCES})
  else()
    add_library(hipcomp SHARED ${HIP_SOURCES} ${CPP_SOURCES})
  endif()
endif()

target_include_directories(hipcomp PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../src>")
target_include_directories(hipcomp PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>")

if (CUDA_BACKEND)
  set_cuda_compilation_options()
endif (CUDA_BACKEND)

if(UNIX)
target_include_directories(hipcomp PUBLIC
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
endif()

include_directories("${hipcomp_SOURCE_DIR}/src")

if (bitcomp_FOUND)
  target_include_directories(hipcomp INTERFACE ${BITCOMP_INCLUDE_DIRS})
  target_link_libraries(hipcomp PRIVATE bitcomp)
endif()

if (gdeflate_FOUND)
  target_include_directories(hipcomp PRIVATE ${GDEFLATE_INCLUDE_DIRS})
  target_link_libraries(hipcomp PRIVATE gdeflate)
endif()

if (ans_FOUND)
  target_include_directories(hipcomp PRIVATE ${ANS_INCLUDE_DIRS})
  target_link_libraries(hipcomp PRIVATE ans)
endif()

if (BUILD_TESTS)
  add_subdirectory("test")
  add_subdirectory("highlevel/test")
  add_subdirectory("lowlevel/test")
endif()
