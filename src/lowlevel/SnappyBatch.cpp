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

#include "hipcomp/snappy.h"

#include "Check.h"
#include "HipUtils.h"
#include "SnappyBatchKernels.h"
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
namespace
{

size_t snappy_get_max_compressed_length(size_t source_bytes)
{
  // This is an estimate from the original snappy library
  return 32 + source_bytes + source_bytes / 6;
}

} // namespace

/******************************************************************************
 *     C-style API calls for BATCHED compression/decompress defined below.
 *****************************************************************************/

hipcompStatus_t hipcompBatchedSnappyDecompressGetTempSize(
    size_t /* num_chunks */,
    size_t /* max_uncompressed_chunk_size */,
    size_t* temp_bytes)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(temp_bytes);

    // Snappy doesn't need any workspace in GPU memory
    *temp_bytes = 0;

  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "hipcompBatchedSnappyDecompressGetTempSize()");
  }

  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedSnappyGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(device_compressed_ptrs);
    CHECK_NOT_NULL(device_compressed_bytes);
    CHECK_NOT_NULL(device_uncompressed_bytes);

    gpu_get_uncompressed_sizes(
        device_compressed_ptrs,
        device_compressed_bytes,
        device_uncompressed_bytes,
        batch_size,
        stream);

  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "hipcompBatchedSnappyGetDecompressSizeAsync()");
  }

  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedSnappyDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const /* temp_ptr */,
    const size_t /* temp_bytes */,
    void* const* device_uncompressed_ptr,
    hipcompStatus_t* device_statuses,
    hipStream_t stream)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(device_compressed_ptrs);
    CHECK_NOT_NULL(device_compressed_bytes);
    CHECK_NOT_NULL(device_uncompressed_bytes);
    CHECK_NOT_NULL(device_uncompressed_ptr);

    gpu_unsnap(
        device_compressed_ptrs,
        device_compressed_bytes,
        device_uncompressed_ptr,
        device_uncompressed_bytes,
        device_statuses,
        device_actual_uncompressed_bytes,
        batch_size,
        stream);

  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "hipcompBatchedSnappyDecompressAsync()");
  }

  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedSnappyCompressGetTempSize(
    const size_t /* batch_size */,
    const size_t /* max_chunk_size */,
    const hipcompBatchedSnappyOpts_t /* format_opts */,
    size_t* const temp_bytes)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(temp_bytes);

    // Snappy doesn't need any workspace in GPU memory
    *temp_bytes = 0;

  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "hipcompBatchedSnappyCompressGetTempSize()");
  }

  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedSnappyCompressGetMaxOutputChunkSize(
    const size_t max_chunk_size,
    const hipcompBatchedSnappyOpts_t /* format_opts */,
    size_t* const max_compressed_size)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(max_compressed_size);

    *max_compressed_size = snappy_get_max_compressed_length(max_chunk_size);

  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "hipcompBatchedSnappyCompressGetOutputSize()");
  }

  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedSnappyCompressAsync(
    const void* const* device_uncompressed_ptr,
    const size_t* device_uncompressed_bytes,
    size_t /*max_uncompressed_chunk_bytes*/,
    size_t batch_size,
    void* /* device_temp_ptr */,
    size_t /* temp_bytes */,
    void* const* device_compressed_ptr,
    size_t* device_compressed_bytes,
    const hipcompBatchedSnappyOpts_t /* format_ops */,
    hipStream_t stream)
{
  try {
    // error check inputs
    CHECK_NOT_NULL(device_uncompressed_ptr);
    CHECK_NOT_NULL(device_uncompressed_bytes);
    CHECK_NOT_NULL(device_compressed_ptr);
    CHECK_NOT_NULL(device_compressed_bytes);

    size_t* device_out_available_bytes = nullptr;
    gpu_snappy_status_s* statuses = nullptr;

    gpu_snap(
        device_uncompressed_ptr,
        device_uncompressed_bytes,
        device_compressed_ptr,
        device_out_available_bytes,
        statuses,
        device_compressed_bytes,
        batch_size,
        stream);

  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "hipcompBatchedSnappyCompressAsync()");
  }

  return hipcompSuccess;
}