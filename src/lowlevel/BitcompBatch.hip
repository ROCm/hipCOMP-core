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

#include "common.h"
#include "hipcomp.h"
#include "hipcomp/bitcomp.h"
#include "type_macros.h"

#ifdef ENABLE_BITCOMP
#include <bitcomp.h>

#define BTCHK(call)                                                            \
  {                                                                            \
    bitcompResult_t err = call;                                                \
    if (BITCOMP_SUCCESS != err) {                                              \
      if (err == BITCOMP_INVALID_PARAMETER)                                    \
        return hipcompErrorInvalidValue;                                        \
      else if (err == BITCOMP_INVALID_COMPRESSED_DATA)                         \
        return hipcompErrorCannotDecompress;                                    \
      else if (err == BITCOMP_INVALID_ALIGNMENT)                               \
        return hipcompErrorCannotDecompress;                                    \
      return hipcompErrorInternal;                                              \
    }                                                                          \
  }

hipcompStatus_t hipcompBatchedBitcompCompressGetMaxOutputChunkSize(
    size_t max_chunk_size,
    hipcompBatchedBitcompFormatOpts format_opts,
    size_t* max_compressed_size)
{
  *max_compressed_size = bitcompMaxBuflen(max_chunk_size);
  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedBitcompCompressAsync(
    const void* const* device_uncompressed_ptrs,
    const size_t* device_uncompressed_bytes,
    size_t, // max_uncompressed_chunk_bytes, not used
    size_t batch_size,
    void*,  // device_temp_ptr, not used
    size_t, // temp_bytes, not used
    void* const* device_compressed_ptrs,
    size_t* device_compressed_bytes,
    const hipcompBatchedBitcompFormatOpts format_opts,
    hipStream_t stream)
{
  // Convert the HIPCOMP type to a BITCOMP type
  bitcompDataType_t dataType;
  switch (format_opts.data_type) {
  case HIPCOMP_TYPE_CHAR:
    dataType = BITCOMP_SIGNED_8BIT;
    break;
  case HIPCOMP_TYPE_USHORT:
    dataType = BITCOMP_UNSIGNED_16BIT;
    break;
  case HIPCOMP_TYPE_SHORT:
    dataType = BITCOMP_SIGNED_16BIT;
    break;
  case HIPCOMP_TYPE_UINT:
    dataType = BITCOMP_UNSIGNED_32BIT;
    break;
  case HIPCOMP_TYPE_INT:
    dataType = BITCOMP_SIGNED_32BIT;
    break;
  case HIPCOMP_TYPE_ULONGLONG:
    dataType = BITCOMP_UNSIGNED_64BIT;
    break;
  case HIPCOMP_TYPE_LONGLONG:
    dataType = BITCOMP_SIGNED_64BIT;
    break;
  default:
    dataType = BITCOMP_UNSIGNED_8BIT;
  }

  // Create a Bitcomp batch handle, associate it to the stream
  bitcompAlgorithm_t algo = static_cast<bitcompAlgorithm_t>(format_opts.algorithm_type);
  bitcompHandle_t plan;
  BTCHK(bitcompCreateBatchPlan(&plan, batch_size, dataType, BITCOMP_LOSSLESS, algo));
  BTCHK(bitcompSetStream(plan, stream));

  // Launch the Bitcomp async batch compression
  BTCHK(bitcompBatchCompressLossless(
      plan,
      device_uncompressed_ptrs,
      device_compressed_ptrs,
      device_uncompressed_bytes,
      device_compressed_bytes));

  // Once launched, the handle can be destroyed
  BTCHK(bitcompDestroyPlan (plan));
  
  return hipcompSuccess;
}

// The Bitcomp batch decompression outputs bitcompResult_t statuses.
// Need to convert them to hipcompStatus_t.
__global__ void convertOutputStatuses (hipcompStatus_t *statuses, size_t batch_size)
{
  static_assert(
      sizeof(hipcompStatus_t) == sizeof(bitcompResult_t),
      "bitcomp and nvcomp statuses must be the same size");
  size_t index = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
  if (index >= batch_size)
      return;
  bitcompResult_t ier = reinterpret_cast<bitcompResult_t *>(statuses)[index];
  hipcompStatus_t hipcomp_err = hipcompSuccess;
  if (ier != BITCOMP_SUCCESS)
  {
      if (ier == BITCOMP_INVALID_PARAMETER)
          hipcomp_err = hipcompErrorInvalidValue;
      else
          hipcomp_err = hipcompErrorCannotDecompress;
  }
  statuses[index] = hipcomp_err;
}

hipcompStatus_t hipcompBatchedBitcompDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t*, // device_compressed_bytes, not used
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const, // device_temp_ptr, not used
    size_t,      // temp_bytes, not used
    void* const* device_uncompressed_ptrs,
    hipcompStatus_t* device_statuses,
    hipStream_t stream)
{
  // Synchronize the stream to make sure the compressed data is visible
  if (hipStreamSynchronize(stream) != hipSuccess)
    return hipcompErrorHipError;

  // Create a Bitcomp batch handle from the compressed data.
  bitcompHandle_t plan;
  BTCHK(bitcompCreateBatchPlanFromCompressedData(&plan, device_compressed_ptrs, batch_size));

  // Associate the handle to the stream
  BTCHK(bitcompSetStream(plan, stream));

  // Launch the Bitcomp async batch decompression with extra checks
  BTCHK(bitcompBatchUncompressCheck(
      plan,
      device_compressed_ptrs,
      device_uncompressed_ptrs,
      device_uncompressed_bytes,
      (bitcompResult_t*)device_statuses));

  // Need a separate kernel to query the actual uncompressed size,
  // as bitcomp doesn't write the uncompressed size during decompression
  BTCHK(bitcompBatchGetUncompressedSizesAsync(
      device_compressed_ptrs,
      device_actual_uncompressed_bytes,
      batch_size,
      stream));

  // Also launch a kernel to convert the output statuses
  const int threads = 512;
  int blocks = (batch_size - 1) / threads + 1;
  convertOutputStatuses<<<blocks, threads, 0, stream>>>(
      device_statuses, batch_size);

  // Once launched, the handle can be destroyed
  BTCHK(bitcompDestroyPlan(plan));
  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedBitcompGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream)
{
  BTCHK(bitcompBatchGetUncompressedSizesAsync(
      device_compressed_ptrs,
      device_uncompressed_bytes,
      batch_size, stream));
  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedBitcompCompressGetTempSize(
    size_t,
    size_t,
    hipcompBatchedBitcompFormatOpts,
    size_t* temp_bytes)
{
  *temp_bytes = 0;
  return hipcompSuccess;
}

hipcompStatus_t hipcompBatchedBitcompDecompressGetTempSize(
    size_t,
    size_t,
    size_t* temp_bytes)
{
  *temp_bytes = 0;
  return hipcompSuccess;
}

#else

hipcompStatus_t hipcompBatchedBitcompCompressGetMaxOutputChunkSize(
    size_t, hipcompBatchedBitcompFormatOpts, size_t*)
{
  return hipcompErrorNotSupported;
}

hipcompStatus_t hipcompBatchedBitcompCompressAsync(
    const void* const*,
    const size_t*,
    size_t,
    size_t,
    void*,
    size_t,
    void* const*,
    size_t*,
    const hipcompBatchedBitcompFormatOpts,
    hipStream_t)
{
  return hipcompErrorNotSupported;
}

hipcompStatus_t hipcompBatchedBitcompDecompressAsync(
    const void* const*,
    const size_t*,
    const size_t*,
    size_t*,
    size_t,
    void* const,
    size_t,
    void* const*,
    hipcompStatus_t*,
    hipStream_t)
{
  return hipcompErrorNotSupported;
}

hipcompStatus_t hipcompBatchedBitcompGetDecompressSizeAsync(
    const void* const*, const size_t*, size_t*, size_t, hipStream_t)
{
  return hipcompErrorNotSupported;
}

hipcompStatus_t hipcompBatchedBitcompCompressGetTempSize(
    size_t, size_t, hipcompBatchedBitcompFormatOpts, size_t*)
{
  return hipcompErrorNotSupported;
}

hipcompStatus_t hipcompBatchedBitcompDecompressGetTempSize(size_t, size_t, size_t*)
{
  return hipcompErrorNotSupported;
}

#endif