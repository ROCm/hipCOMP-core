/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
// Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights
// reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "hipcomp/helpers.h"
#include "lowlevel/DeflateBatchKernels.h"
// #include "deflate/compression.cuh"
#include "Check.h"
#include "HipUtils.h"
#include "deflate/decompression.cuh"

namespace hipcomp {

/**
 * \brief Returns an estimate for the uncompressed size of each chunk.
 *
 * Guess an initial maximum uncompressed block size. We estimate the compression
 * factor is two and round up to the next multiple of 4096 bytes. Note that this
 * estimate is the same as for DEFLATE's gpuinflate.
 *
 * \note The estimate we use is from cuDF:
 * https://github.com/rapidsai/cudf/blob/f592e9c4bfcc2d8e887ad5f96e5167ee0ee2c73a/cpp/src/io/avro/reader_impl.cu#L207
 */
__global__ void get_uncompressed_sizes_estimate_kernel(
    const uint64_t *__restrict__ device_in_bytes,
    uint64_t *__restrict__ device_out_bytes, int num_chunks) {
  int chunk_id = threadIdx.x + blockDim.x * blockIdx.x;
  if (chunk_id < num_chunks) {
    device_out_bytes[chunk_id] =
        device_in_bytes[chunk_id] * 2 + (device_in_bytes[chunk_id] * 2) % 4096;
  }
}

/**
 * @brief Deflate/Gzip decompression kernel
 *
 * blockDim {DECOMP_THREADS_PER_BLOCK,1,1}
 *
 * @param[in] inputs Source & destination information per block
 * @param[out] outputs Decompression status per block
 **/
__global__ void __launch_bounds__(DECOMP_THREADS_PER_BLOCK)
    gpu_inflate_kernel(const void *const *__restrict__ device_in_ptr,
                       const uint64_t *__restrict__ device_in_bytes,
                       void *const *__restrict__ device_out_ptr,
                       const uint64_t *__restrict__ device_out_available_bytes,
                       hipcompStatus_t *const __restrict__ outputs,
                       uint64_t *__restrict__ device_out_bytes,
                       uint32_t *const __restrict__ device_reserved,
                       bool parse_hdr) {
  const int ix_chunk = blockIdx.x;
  gzip::do_inflate(
      reinterpret_cast<const uint8_t *>(device_in_ptr[ix_chunk]),
      device_in_bytes[ix_chunk],
      reinterpret_cast<uint8_t *>(device_out_ptr[ix_chunk]),
      device_out_available_bytes ? device_out_available_bytes[ix_chunk] : 0,
      outputs ? &outputs[ix_chunk] : nullptr,
      device_out_bytes ? &device_out_bytes[ix_chunk] : nullptr,
      device_reserved ? &device_reserved[ix_chunk] : nullptr, parse_hdr);
}

void gpu_inflate(const void *const *device_in_ptr,
                 const size_t *device_in_bytes, void *const *device_out_ptr,
                 const size_t *device_out_available_bytes,
                 hipcompStatus_t *outputs, size_t *device_out_bytes,
                 uint32_t *const device_reserved, int count, bool parse_hdr,
                 hipStream_t stream) {
  uint32_t count32 = (count > 0) ? count : 0;
  dim3 dim_block(DECOMP_THREADS_PER_BLOCK, 1);
  dim3 dim_grid(count32,
                1); // TODO: Check max grid dimensions vs max expected count

  gpu_inflate_kernel<<<dim_grid, dim_block, 0, stream>>>(
      device_in_ptr, device_in_bytes, device_out_ptr,
      device_out_available_bytes, outputs, device_out_bytes, device_reserved,
      parse_hdr);
  HipUtils::check_last_error(
      "Failed to launch Gzip/Deflate decompression HIP kernel gpu_inflate");
}

void gpu_get_uncompressed_sizes_estimate(const size_t *device_in_bytes,
                                         size_t *device_out_bytes, int count,
                                         hipStream_t stream) {
  dim3 dim_block(warpsize, 1); // only a single thread is active in any case
  dim3 dim_grid(count, 1);

  get_uncompressed_sizes_estimate_kernel<<<dim_grid, dim_block, 0, stream>>>(
      device_in_bytes, device_out_bytes, count);
  HipUtils::check_last_error(
      "Failed to run Gzip/Deflate kernel gpu_get_uncompressed_sizes");
}

} // namespace hipcomp
