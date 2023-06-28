/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef HIPCOMP_HIPUTILS_H
#define HIPCOMP_HIPUTILS_H

#include "hip/hip_runtime.h"

#include <string>

namespace hipcomp
{

enum CopyDirection {
  HOST_TO_DEVICE = hipMemcpyHostToDevice,
  DEVICE_TO_HOST = hipMemcpyDeviceToHost,
  DEVICE_TO_DEVICE = hipMemcpyDeviceToDevice
};

class HipUtils
{
public:
  /**
   * @brief Convert hip errors into exceptions. Will throw an exception
   * unless `err == hipSuccess`.
   *
   * @param err The error.
   * @param msg The message to attach to the exception.
   */
  static void check(const hipError_t err, const std::string& msg = "");

  static void sync(hipStream_t stream);

  static void check_last_error(const std::string& msg = "");

  /**
   * @brief Perform checked asynchronous memcpy.
   *
   * @tparam T The data type.
   * @param dst The destination address.
   * @param src The source address.
   * @param count The number of elements to copy.
   * @param kind The direction of the copy.
   * @param stream THe stream to operate on.
   */
  template <typename T>
  static void copy_async(
      T* const dst,
      const T* const src,
      const size_t count,
      const CopyDirection kind,
      hipStream_t stream)
  {
    check(
        hipMemcpyAsync(dst, src, sizeof(T) * count,
          static_cast<hipMemcpyKind>(kind), stream),
        "HipUtils::copy_async(dst, src, count, kind, stream)");
  }

  /**
   * @brief Perform a synchronous memcpy.
   *
   * @tparam T The data type.
   * @param dst The destination address.
   * @param src The source address.
   * @param count The number of elements to copy.
   * @param kind The direction of the copy.
   */
  template <typename T>
  static void copy(
      T* const dst,
      const T* const src,
      const size_t count,
      const CopyDirection kind)
  {
    check(
        hipMemcpy(dst, src, sizeof(T) * count, static_cast<hipMemcpyKind>(kind)),
        "HipUtils::copy(dst, src, count, kind)");
  }

  static bool is_device_pointer(const void* ptr);

  template <typename T>
  static T* device_pointer(T* const ptr)
  {
    return reinterpret_cast<T*>(void_device_pointer(ptr));
  }

private:
  static const void* void_device_pointer(const void* ptr);
  static void* void_device_pointer(void* ptr);
};

} // namespace hipcomp

#endif