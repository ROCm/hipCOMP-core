/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "hipcomp/lz4.h"

#include "Check.h"
#include "HipUtils.h"
#include "LZ4CompressionKernels.h"
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

using namespace hipcomp;
using namespace hipcomp::lowlevel;

hipcompStatus_t hipcompBatchedLZ4DecompressGetTempSize(
    const size_t num_chunks,
    const size_t max_uncompressed_chunk_size,
    size_t* const temp_bytes)
{
  CHECK_NOT_NULL(temp_bytes);

  try {
    *temp_bytes
        = lz4DecompressComputeTempSize(num_chunks, max_uncompressed_chunk_size);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "hipcompBatchedLZ4DecompressGetTempSize()");
  }

  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedLZ4DecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    hipcompStatus_t* device_statuses,
    hipStream_t stream)
{
  // NOTE: if we start using `max_uncompressed_chunk_bytes`, we need to check
  // to make sure it is not zero, as we have notified users to supply zero if
  // they are not finding the maximum size.

  try {
    lz4BatchDecompress(
        HipUtils::device_pointer(
            reinterpret_cast<const uint8_t* const*>(device_compressed_ptrs)),
        HipUtils::device_pointer(device_compressed_bytes),
        HipUtils::device_pointer(device_uncompressed_bytes),
        batch_size,
        HipUtils::device_pointer(device_temp_ptr),
        temp_bytes,
        HipUtils::device_pointer(
            reinterpret_cast<uint8_t* const*>(device_uncompressed_ptrs)),
        device_actual_uncompressed_bytes ? HipUtils::device_pointer(device_actual_uncompressed_bytes) : nullptr,
        device_statuses ? HipUtils::device_pointer(device_statuses) : nullptr,
        stream);

  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "hipcompBatchedLZ4DecompressAsync()");
  }

  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedLZ4GetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream)
{
  CHECK_NOT_NULL(device_compressed_ptrs);
  CHECK_NOT_NULL(device_compressed_bytes);
  CHECK_NOT_NULL(device_uncompressed_bytes);

  try {
    lz4BatchGetDecompressSizes(
        HipUtils::device_pointer(
            reinterpret_cast<const uint8_t* const*>(device_compressed_ptrs)),
        HipUtils::device_pointer(device_compressed_bytes),
        HipUtils::device_pointer(device_uncompressed_bytes),
        batch_size,
        stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "hipcompBatchedLZ4GetDecompressSizeAsync()");
  }

  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedLZ4CompressGetTempSize(
    const size_t batch_size,
    const size_t max_chunk_size,
    const hipcompBatchedLZ4Opts_t /* format_opts */,
    size_t* const temp_bytes)
{
  CHECK_NOT_NULL(temp_bytes);

  try {
    *temp_bytes = lz4BatchCompressComputeTempSize(max_chunk_size, batch_size);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "hipcompBatchedLZ4CompressGetTempSize()");
  }

  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedLZ4CompressGetMaxOutputChunkSize(
    const size_t max_chunk_size,
    const hipcompBatchedLZ4Opts_t /* format_opts */,
    size_t* const max_compressed_size)
{
  CHECK_NOT_NULL(max_compressed_size);

  try {
    *max_compressed_size = lz4ComputeMaxSize(max_chunk_size);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "hipcompBatchedLZ4CompressGetOutputSize()");
  }

  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedLZ4CompressAsync(
    const void* const* const device_uncompressed_ptrs,
    const size_t* const device_uncompressed_bytes,
    const size_t max_uncompressed_chunk_size,
    const size_t batch_size,
    void* const device_temp_ptr,
    const size_t temp_bytes,
    void* const* const device_compressed_ptrs,
    size_t* const device_compressed_bytes,
    const hipcompBatchedLZ4Opts_t format_opts,
    hipStream_t stream)
{
  // NOTE: if we start using `max_uncompressed_chunk_bytes`, we need to check
  // to make sure it is not zero, as we have notified users to supply zero if
  // they are not finding the maximum size.

  try {
    lz4BatchCompress(
        HipUtils::device_pointer(
            reinterpret_cast<const uint8_t* const*>(device_uncompressed_ptrs)),
        HipUtils::device_pointer(device_uncompressed_bytes),
        max_uncompressed_chunk_size,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        HipUtils::device_pointer(
            reinterpret_cast<uint8_t* const*>(device_compressed_ptrs)),
        HipUtils::device_pointer(device_compressed_bytes),
        format_opts.data_type,
        stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "hipcompBatchedLZ4CompressAsync()");
  }

  return hipcompSuccess;
}