/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

#include "HipUtils.h"

#include <sstream>
#include <stdexcept>

namespace hipcomp
{

namespace
{
std::string to_string(const void* const ptr)
{
  std::ostringstream oss;
  oss << ptr;
  return oss.str();
}
} // namespace

void HipUtils::check(const hipError_t err, const std::string& msg)
{
  if (err != hipSuccess) {
    std::string errorStr(
        "Encountered Hip Error: " + std::to_string(err) + ": '"
        + std::string(hipGetErrorString(err)) + "'");
    if (!msg.empty()) {
      errorStr += ": " + msg;
    }
    errorStr += ".";

    throw std::runtime_error(errorStr);
  }
}

void HipUtils::sync(hipStream_t stream)
{
  check(hipStreamSynchronize(stream), "Failed to sync with stream");
}

void HipUtils::check_last_error(const std::string& msg)
{
  check(hipGetLastError(), msg);
}

const void* HipUtils::void_device_pointer(const void* const ptr)
{
  hipPointerAttribute_t attr;
  check(
      hipPointerGetAttributes(&attr, ptr),
      "Failed to get pointer "
      "attributes for pointer: "
          + to_string(ptr));

  if (!attr.devicePointer) {
    throw std::runtime_error(
        "Memory location is not accessible by the "
        "current GPU: "
        + to_string(ptr));
  }

  return attr.devicePointer;
}

bool HipUtils::is_device_pointer(const void* const ptr)
{
  hipPointerAttribute_t attr;

  hipError_t err = hipPointerGetAttributes(&attr, ptr);

  #if defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__)
  if (err == hipErrorInvalidValue) {
    int cuda_version;
    check(
        hipRuntimeGetVersion(&cuda_version),
        "Failed to get runtime "
        "verison.");

    if (cuda_version < 11000) {
      // error is normal for non-device memory -- clear the error and return
      // false
      (void)hipGetLastError();
      return false;
    }
  }
  #else
  if (err == hipErrorInvalidValue) {
    // error is normal for non-device memory -- clear the error and return
    // false
    static_cast<void>(hipGetLastError());
    return false;
  }
  #endif

  // if we continue, make sure we successfully got pointer information
  check(
      err,
      "Failed to get pointer "
      "attributes for pointer: "
          + to_string(ptr));

  return attr.type == hipMemoryTypeDevice;
}

void* HipUtils::void_device_pointer(void* const ptr)
{
  hipPointerAttribute_t attr;
  // we don't need to worry about the difference between cuda 10 and cuda 11
  // here, as if it's not a device pointer, we want throw an exception either
  // way.
  check(
      hipPointerGetAttributes(&attr, ptr),
      "Failed to get pointer "
      "attributes for pointer: "
          + to_string(ptr));

  if (!attr.devicePointer) {
    throw std::runtime_error(
        "Memory location is not accessible by the "
        "current GPU: "
        + to_string(ptr));
  }

  return attr.devicePointer;
}

} // namespace hipcomp
