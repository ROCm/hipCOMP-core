/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
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

#include "hipcomp/ans.h"

#include "Check.h"
#include "HipUtils.h"
#include "common.h"
#include "hipcomp.h"
#include "hipcomp.hpp"
#include "type_macros.h"

#include <cassert>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <vector>

#ifdef ENABLE_ANS
#include "ans.h"
#endif

using namespace hipcomp;

#define MAYBE_UNUSED(x) (void)(x)

hipcompStatus_t hipcompBatchedANSDecompressGetTempSize(
    const size_t num_chunks,
    const size_t max_uncompressed_chunk_size,
    size_t* const temp_bytes)
{
#ifdef ENABLE_ANS
  CHECK_NOT_NULL(temp_bytes);
  ans::decompressGetTempSize(num_chunks, max_uncompressed_chunk_size, temp_bytes);
  return hipcompSuccess;
#else
  (void)num_chunks;
  (void)max_uncompressed_chunk_size;
  (void)temp_bytes;
  std::cerr << "ERROR: hipcomp configured without GPU ANS support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return hipcompErrorNotSupported;
#endif
}

hipcompStatus_t hipcompBatchedANSDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    const size_t temp_bytes,
    void* const* device_uncompressed_ptr,
    hipcompStatus_t* device_statuses,
    hipStream_t stream)
{
#ifdef ENABLE_ANS
  try {
    ans::decompressAsync(
      HipUtils::device_pointer(device_compressed_ptrs),
      HipUtils::device_pointer(device_compressed_bytes),
      HipUtils::device_pointer(device_uncompressed_bytes),
      device_actual_uncompressed_bytes ? HipUtils::device_pointer(device_actual_uncompressed_bytes) : nullptr,
      0, batch_size, device_temp_ptr, temp_bytes,
      HipUtils::device_pointer(device_uncompressed_ptr),
      device_statuses ? HipUtils::device_pointer(device_statuses) : nullptr,
      stream);
  } catch (const std::exception& e) {
     return Check::exception_to_error(e, "hipcompBatchedANSDecompressAsync()");
  }
  return hipcompSuccess;
#else
  (void)device_compressed_ptrs;
  (void)device_compressed_bytes;
  (void)device_uncompressed_bytes;
  (void)device_actual_uncompressed_bytes;
  (void)batch_size;
  (void)device_temp_ptr;
  (void)temp_bytes;
  (void)device_uncompressed_ptr;
  (void)device_statuses;
  (void)stream;
  std::cerr << "ERROR: hipcomp configured without GPU ANS support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return hipcompErrorNotSupported;
#endif
}

hipcompStatus_t hipcompBatchedANSCompressGetTempSize(
    size_t batch_size,
    size_t max_chunk_size,
    hipcompBatchedANSOpts_t /* format_opts */,
    size_t* temp_bytes)
{
#ifdef ENABLE_ANS
  CHECK_NOT_NULL(temp_bytes);
  ans::compressGetTempSize(batch_size, max_chunk_size, temp_bytes);
  return hipcompSuccess;
#else
  (void)batch_size;
  (void)max_chunk_size;
  (void)temp_bytes;
  std::cerr << "ERROR: hipcomp configured without GPU ANS support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return hipcompErrorNotSupported;
#endif
}

hipcompStatus_t hipcompBatchedANSCompressGetMaxOutputChunkSize(
    size_t max_chunk_size,
    hipcompBatchedANSOpts_t /* format_opts */,
    size_t* max_compressed_size)
{
#ifdef ENABLE_ANS
  CHECK_NOT_NULL(max_compressed_size);
  ans::compressGetMaxOutputChunkSize(max_chunk_size, max_compressed_size);
  return hipcompSuccess;
#else
  (void)max_chunk_size;
  (void)max_compressed_size;
  std::cerr << "ERROR: hipcomp configured without GPU ANS support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return hipcompErrorNotSupported;
#endif
}

hipcompStatus_t hipcompBatchedANSCompressAsync(
    const void* const* device_uncompressed_ptr,
    const size_t* device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_compressed_ptr,
    size_t* device_compressed_bytes,
    hipcompBatchedANSOpts_t format_opts,
    hipStream_t stream)
{
#ifdef ENABLE_ANS
  assert(format_opts.type == hipcompANSType_t::hipcomp_rANS);
  MAYBE_UNUSED(format_opts);
  ans::ansType_t ans_type = ans::ansType_t::rANS;

  try {
    ans::compressAsync(
        ans_type,
        HipUtils::device_pointer(device_uncompressed_ptr),
        HipUtils::device_pointer(device_uncompressed_bytes),
        max_uncompressed_chunk_bytes,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        HipUtils::device_pointer(device_compressed_ptr),
        HipUtils::device_pointer(device_compressed_bytes),
        stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "hipcompBatchedANSCompressAsync()");
  }
  return hipcompSuccess;
#else
  (void)device_uncompressed_ptr;
  (void)device_uncompressed_bytes;
  (void)max_uncompressed_chunk_bytes;
  (void)batch_size;
  (void)device_temp_ptr;
  (void)temp_bytes;
  (void)device_compressed_ptr;
  (void)device_compressed_bytes;
  (void)format_opts;
  (void)stream;
  std::cerr << "ERROR: hipcomp configured without GPU ANS support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return hipcompErrorNotSupported;
#endif
}

hipcompStatus_t hipcompBatchedANSGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* /* device_compressed_bytes */,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream) {
#ifdef ENABLE_ANS
  ans::getDecompressSizeAsync(
      device_compressed_ptrs,
      device_uncompressed_bytes,
      batch_size,
      stream);
  return hipcompSuccess;
#else
  (void)device_compressed_ptrs;
  (void)device_uncompressed_bytes;
  (void)batch_size;
  (void)stream;
  std::cerr << "ERROR: hipcomp configured without GPU ANS support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return hipcompErrorNotSupported;
#endif
}
