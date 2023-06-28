/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef HIPCOMP_API_HPP
#define HIPCOMP_API_HPP

#include "hipcomp.h"
#include "hipcomp/lz4.h"

#include <cstdint>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace hipcomp
{

/******************************************************************************
 * CLASSES ********************************************************************
 *****************************************************************************/

/**
 * @brief The top-level exception throw by hipcomp C++ methods.
 */
class HipCompException : public std::runtime_error
{
public:
  /**
   * @brief Create a new HipCompException.
   *
   * @param err The error associated with the exception.
   * @param msg The error message.
   */
  HipCompException(hipcompStatus_t err, const std::string& msg) :
      std::runtime_error(msg + " : code=" + std::to_string(err) + "."),
      m_err(err)
  {
    // do nothing
  }

  hipcompStatus_t get_error() const
  {
    return m_err;
  }

private:
  hipcompStatus_t m_err;
};

/**
 * @brief Top-level compressor class. This class takes in data on the device,
 * and compresses it to another location on the device.
 */
class Compressor
{
public:
  /**
   * @brief Virtual destructor.
   */
  virtual ~Compressor() = default;

  virtual void
  configure(const size_t in_bytes, size_t* temp_bytes, size_t* out_bytes)
      = 0;

  /**
   * @brief Launch asynchronous compression. If the `out_bytes` is pageable
   * memory, this method will block.
   *
   * @param temp_ptr The temporary workspace on the device.
   * @param temp_bytes The size of the temporary workspace.
   * @param out_ptr The output location the the device (for compressed data).
   * @param out_bytes The size of the output location on the device on input,
   * and the size of the compressed data on output.
   * @param stream The stream to operate on.
   *
   * @throw HipCompException If compression fails to launch on the stream.
   */
  virtual void compress_async(
      const void* in_ptr,
      const size_t in_bytes,
      void* temp_ptr,
      const size_t temp_bytes,
      void* out_ptr,
      size_t* out_bytes,
      hipStream_t stream)
      = 0;
};

/**
 * @brief Top-level decompress class. The compression type is read from the
 * metadata at the start of the compressed data.
 */
class Decompressor
{

public:
  virtual ~Decompressor() = default;

  virtual void configure(
      const void* in_ptr,
      const size_t in_bytes,
      size_t* temp_bytes,
      size_t* out_bytes,
      hipStream_t stream)
      = 0;

  virtual void decompress_async(
      const void* in_ptr,
      const size_t in_bytes,
      void* temp_ptr,
      const size_t temp_bytes,
      void* out_ptr,
      const size_t out_bytes,
      hipStream_t stream)
      = 0;
};

/******************************************************************************
 * INLINE DEFINITIONS AND HELPER FUNCTIONS ************************************
 *****************************************************************************/

template <typename T>
inline hipcompType_t TypeOf()
{
  if (std::is_same<T, int8_t>::value) {
    return HIPCOMP_TYPE_CHAR;
  } else if (std::is_same<T, uint8_t>::value) {
    return HIPCOMP_TYPE_UCHAR;
  } else if (std::is_same<T, int16_t>::value) {
    return HIPCOMP_TYPE_SHORT;
  } else if (std::is_same<T, uint16_t>::value) {
    return HIPCOMP_TYPE_USHORT;
  } else if (std::is_same<T, int32_t>::value) {
    return HIPCOMP_TYPE_INT;
  } else if (std::is_same<T, uint32_t>::value) {
    return HIPCOMP_TYPE_UINT;
  } else if (std::is_same<T, int64_t>::value) {
    return HIPCOMP_TYPE_LONGLONG;
  } else if (std::is_same<T, uint64_t>::value) {
    return HIPCOMP_TYPE_ULONGLONG;
  } else {
    throw HipCompException(
        hipcompErrorNotSupported, "hipcomp does not support the given type.");
  }
}

inline void throwExceptionIfError(hipcompStatus_t error, const std::string& msg)
{
  if (error != hipcompSuccess) {
    throw HipCompException(error, msg);
  }
}


} // namespace hipcomp

#endif