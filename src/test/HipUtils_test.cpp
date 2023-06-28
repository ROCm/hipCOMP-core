/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// MIT License
//
// Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define CATCH_CONFIG_MAIN

#include "tests/catch.hpp"
#include "HipUtils.h"

#include "hip/hip_runtime.h"

#ifndef HIP_RT_CALL
#define HIP_RT_CALL(call)                                                     \
  {                                                                            \
    hipError_t hipStatus = call;                                             \
    if (hipSuccess != hipStatus) {                                           \
      fprintf(                                                                 \
          stderr,                                                              \
          "ERROR: HIP RT call \"%s\" in line %d of file %s failed with %s "   \
          "(%d).\n",                                                           \
          #call,                                                               \
          __LINE__,                                                            \
          __FILE__,                                                            \
          hipGetErrorString(hipStatus),                                      \
          hipStatus);                                                         \
      abort();                                                                 \
    }                                                                          \
  }
#endif

using namespace hipcomp;

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST_CASE("IsDevicePointerTest", "[small]")
{
  // check a device pointer - true
  size_t* dev_ptr;
  HIP_RT_CALL(hipMalloc((void**)&dev_ptr, sizeof(*dev_ptr)));
  REQUIRE(HipUtils::is_device_pointer(dev_ptr));
  HIP_RT_CALL(hipFree(dev_ptr));

  // check a uvm pointer - false
  size_t* managed_ptr;
  HIP_RT_CALL(hipMallocManaged((void**)&managed_ptr, sizeof(*managed_ptr)));
  REQUIRE(!HipUtils::is_device_pointer(managed_ptr));
  HIP_RT_CALL(hipFree(managed_ptr));

  // check a pinned pointer - false
  size_t* pinned_ptr;
  HIP_RT_CALL(hipHostMalloc((void**)&pinned_ptr, sizeof(*pinned_ptr)));
  REQUIRE(!HipUtils::is_device_pointer(pinned_ptr));
  HIP_RT_CALL(hipHostFree(pinned_ptr));

  // check an unregistered pointer - false
  size_t unregistered;
  REQUIRE(!HipUtils::is_device_pointer(&unregistered));

  // check a null pointer - should be false
  REQUIRE(!HipUtils::is_device_pointer(nullptr));
}

TEST_CASE("DevicePointerTest", "[small]")
{
  // check a device pointer - should be equal
  size_t* dev_ptr;
  HIP_RT_CALL(hipMalloc((void**)&dev_ptr, sizeof(*dev_ptr)));
  REQUIRE(HipUtils::device_pointer(dev_ptr) == dev_ptr);
  HIP_RT_CALL(hipFree(dev_ptr));

  // check a uvm pointer - should succeed and return a device pointer
  size_t* managed_ptr;
  HIP_RT_CALL(hipMallocManaged((void**)&managed_ptr, sizeof(*managed_ptr)));
  size_t* managed_dev_ptr = HipUtils::device_pointer(managed_ptr);
  HIP_RT_CALL(hipMemset(managed_dev_ptr, 0, sizeof(*managed_dev_ptr)));
  HIP_RT_CALL(hipFree(managed_ptr));

  // check a pinned pointer - should succeed and return a device pointer
  size_t* pinned_ptr;
  HIP_RT_CALL(hipHostMalloc((void**)&pinned_ptr, sizeof(*pinned_ptr)));
  size_t* pinned_dev_ptr = HipUtils::device_pointer(pinned_ptr);
  HIP_RT_CALL(hipMemset(pinned_dev_ptr, 0, sizeof(*pinned_dev_ptr)));
  HIP_RT_CALL(hipHostFree(pinned_ptr));

  // check an unregistered pointer - should throw an exception
  try {
    size_t unregistered;
    HipUtils::device_pointer(&unregistered);
    REQUIRE(false); // unreachable
  } catch (const std::exception&) {
    // pass
  }

  // check a null pointer - should throw an exception
  try {
    HipUtils::device_pointer(static_cast<void*>(nullptr));
  } catch (const std::exception&) {
    // pass
  }
}
