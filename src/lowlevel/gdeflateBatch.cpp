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

#include "hipcomp/gdeflate.h"

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

#ifdef ENABLE_GDEFLATE
#include "gdeflate.h"
#include "gdeflateKernels.h"
#endif

using namespace hipcomp;

#ifdef ENABLE_GDEFLATE
gdeflate::gdeflate_compression_algo getGdeflateEnumFromFormatOpts(hipcompBatchedGdeflateOpts_t format_opts) {
  gdeflate::gdeflate_compression_algo algo;
  switch(format_opts.algo) {
    case (0) :
      algo = gdeflate::HIGH_THROUGHPUT;
      break;
    case(1) :
      algo = gdeflate::HIGH_COMPRESSION;
      break;
    case(2) :
      algo = gdeflate::ENTROPY_ONLY;
      break;
    default :
      throw std::invalid_argument("Invalid format_opts.algo value (not 0, 1 or 2)");
  }
  return algo;
}
#endif

hipcompStatus_t hipcompBatchedGdeflateDecompressGetTempSize(
    const size_t num_chunks,
    const size_t max_uncompressed_chunk_size,
    size_t* const temp_bytes)
{
#ifdef ENABLE_GDEFLATE
  CHECK_NOT_NULL(temp_bytes);

  try {
    gdeflate::decompressGetTempSize(num_chunks, max_uncompressed_chunk_size, temp_bytes);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "hipcompBatchedGdeflateDecompressGetTempSize()");
  }

  return hipcompSuccess;
#else
  (void)num_chunks;
  (void)max_uncompressed_chunk_size;
  (void)temp_bytes;
  std::cerr << "ERROR: hipcomp configured without gdeflate support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return hipcompErrorNotSupported;
#endif
}

hipcompStatus_t hipcompBatchedGdeflateDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    hipcompStatus_t* device_status_ptrs,
    hipStream_t stream)
{
#ifdef ENABLE_GDEFLATE
  // NOTE: if we start using `max_uncompressed_chunk_bytes`, we need to check
  // to make sure it is not zero, as we have notified users to supply zero if
  // they are not finding the maximum size.

  try {
    // Use device_status_ptrs as temp space to store gdeflate statuses
    static_assert(sizeof(hipcompStatus_t) == sizeof(gdeflate::gdeflateStatus_t),
        "Mismatched sizes of hipcompStatus_t and gdeflateStatus_t");
    auto device_statuses = reinterpret_cast<gdeflate::gdeflateStatus_t*>(device_status_ptrs);

    // Run the decompression kernel
    gdeflate::decompressAsync(device_compressed_ptrs, device_compressed_bytes,
        device_uncompressed_bytes, device_actual_uncompressed_bytes,
        0, batch_size, device_temp_ptr, temp_bytes,
        device_uncompressed_ptrs, device_statuses, stream);

    // Launch a kernel to convert the output statuses
    if(device_status_ptrs) convertGdeflateOutputStatuses(device_status_ptrs, batch_size, stream);

  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "hipcompBatchedGdeflateDecompressAsync()");
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
  (void)device_uncompressed_ptrs;
  (void)device_status_ptrs;
  (void)stream;
  std::cerr << "ERROR: hipcomp configured without gdeflate support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return hipcompErrorNotSupported;
#endif
}

hipcompStatus_t hipcompBatchedGdeflateGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream) {
#ifdef ENABLE_GDEFLATE
  try {
    gdeflate::getDecompressSizeAsync(device_compressed_ptrs, device_compressed_bytes,
        device_uncompressed_bytes, batch_size, stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "hipcompBatchedGdeflateDecompressAsync()");
  }

  return hipcompSuccess;
#else
  (void)device_compressed_ptrs;
  (void)device_compressed_bytes;
  (void)device_uncompressed_bytes;
  (void)batch_size;
  (void)stream;
  std::cerr << "ERROR: hipcomp configured without gdeflate support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return hipcompErrorNotSupported;
#endif
}

hipcompStatus_t hipcompBatchedGdeflateCompressGetTempSize(
    const size_t batch_size,
    const size_t max_chunk_size,
    hipcompBatchedGdeflateOpts_t format_opts,
    size_t* const temp_bytes)
{
#ifdef ENABLE_GDEFLATE
  CHECK_NOT_NULL(temp_bytes);


  try {
    gdeflate::gdeflate_compression_algo algo = getGdeflateEnumFromFormatOpts(format_opts);
    gdeflate::compressGetTempSize(batch_size, max_chunk_size, temp_bytes, algo);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "hipcompBatchedGdeflateCompressGetTempSize()");
  }

  return hipcompSuccess;
#else
  (void)batch_size;
  (void)max_chunk_size;
  (void)format_opts;
  (void)temp_bytes;
  std::cerr << "ERROR: hipcomp configured without gdeflate support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return hipcompErrorNotSupported;
#endif
}

hipcompStatus_t hipcompBatchedGdeflateCompressGetMaxOutputChunkSize(
    size_t max_chunk_size,
    hipcompBatchedGdeflateOpts_t /* format_opts */,
    size_t* max_compressed_size)
{
#ifdef ENABLE_GDEFLATE
  CHECK_NOT_NULL(max_compressed_size);

  try {
    gdeflate::compressGetMaxOutputChunkSize(max_chunk_size, max_compressed_size);
  } catch (const std::exception& e) {
    return Check::exception_to_error(
        e, "hipcompBatchedGdeflateCompressGetOutputSize()");
  }

  return hipcompSuccess;
#else
  (void)max_chunk_size;
  (void)max_compressed_size;
  std::cerr << "ERROR: hipcomp configured without gdeflate support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return hipcompErrorNotSupported;
#endif
}

hipcompStatus_t hipcompBatchedGdeflateCompressAsync(
    const void* const* const device_in_ptrs,
    const size_t* const device_in_bytes,
    const size_t max_uncompressed_chunk_size,
    const size_t batch_size,
    void* const temp_ptr,
    const size_t temp_bytes,
    void* const* const device_out_ptrs,
    size_t* const device_out_bytes,
    hipcompBatchedGdeflateOpts_t format_opts,
    hipStream_t stream)
{
#ifdef ENABLE_GDEFLATE
  try {
    gdeflate::gdeflate_compression_algo algo = getGdeflateEnumFromFormatOpts(format_opts);
    gdeflate::compressAsync(device_in_ptrs, device_in_bytes, max_uncompressed_chunk_size,
        batch_size, temp_ptr, temp_bytes, device_out_ptrs, device_out_bytes, algo, stream);
  } catch (const std::exception& e) {
    return Check::exception_to_error(e, "hipcompBatchedGdeflateCompressAsync()");
  }

  return hipcompSuccess;
#else
  (void)device_in_ptrs;
  (void)device_in_bytes;
  (void)max_uncompressed_chunk_size;
  (void)batch_size;
  (void)temp_ptr;
  (void)temp_bytes;
  (void)device_out_ptrs;
  (void)device_out_bytes;
  (void)format_opts;
  (void)stream;
  std::cerr << "ERROR: hipcomp configured without gdeflate support\n"
            << "Please check the README for configuration instructions" << std::endl;
  return hipcompErrorNotSupported;
#endif
}
