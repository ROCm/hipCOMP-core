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
// Modifications Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights
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
#include "lowlevel/ZstdBatchKernels.h"
// #include "zstd/compression.cuh"
#include "Check.h"
#include "HipUtils.h"
#include "zstd/hipv00/decompression.cuh"

#if DEBUGLEVEL > 0
#include <string>
#include <vector>
#endif

namespace hipcomp {

/**
 * @brief ZSTD decompression kernel
 *
 * blockDim {DECOMP_THREADS_PER_BLOCK,1,1}
 *
 * @param[in] device_in_ptr Source buffer pointers
 * @param[in] device_in_bytes Source buffer sizes
 * @param[out] device_out_ptr Destination buffer pointers
 * @param[in] device_out_available_bytes Destination buffer sizes
 * @param[out] outputs Decompression status per block
 * @param[out] device_out_bytes Actual decompressed sizes
 **/
template <int warpsize>
__global__ void __launch_bounds__(zstd::DECOMP_WARPS_PER_BLOCK *warpsize)
    gpu_zstd_decompress_kernel(
        const void *const *__restrict__ device_in_ptr,
        const uint64_t *__restrict__ device_in_bytes,
        void *const *__restrict__ device_out_ptr,
        const uint64_t *__restrict__ device_out_available_bytes,
        hipcompStatus_t *const __restrict__ outputs,
        uint64_t *__restrict__ device_out_bytes) {
  const int ix_chunk = blockIdx.x;
  zstd::do_decompress(
      reinterpret_cast<const uint8_t *>(device_in_ptr[ix_chunk]),
      device_in_bytes[ix_chunk],
      reinterpret_cast<uint8_t *>(device_out_ptr[ix_chunk]),
      device_out_available_bytes ? device_out_available_bytes[ix_chunk] : 0,
      outputs ? &outputs[ix_chunk] : nullptr,
      device_out_bytes ? &device_out_bytes[ix_chunk] : nullptr);
}

void gpu_zstd_decompress(const void *const *device_in_ptr,
                         const size_t *device_in_bytes,
                         void *const *device_out_ptr,
                         const size_t *device_out_available_bytes,
                         hipcompStatus_t *outputs, size_t *device_out_bytes,
                         int count, hipStream_t stream) {
  uint32_t count32 = (count > 0) ? count : 0;

  HIPCOMP_EXECUTE_WARPSIZE_DEPENDENT_CODE(
      -1, constexpr int WS = HIPCOMP_WARPSIZE;

      dim3 dim_block(zstd::DECOMP_WARPS_PER_BLOCK * WS, 1);
      dim3 dim_grid(count);

      gpu_zstd_decompress_kernel<WS><<<dim_grid, dim_block, 0, stream>>>(
          device_in_ptr, device_in_bytes, device_out_ptr,
          device_out_available_bytes, outputs, device_out_bytes);)

  HipUtils::check_last_error(
      "Failed to launch ZSTD decompression HIP kernel gpu_zstd_decompress");
}

/**
 * @brief Kernel to get the uncompressed size for each ZSTD chunk.
 *
 * This kernel uses ZSTD's frame header reading to determine the actual
 * uncompressed size, which is much more accurate than estimation.
 *
 * @param device_in_ptr Pointer to array of compressed data pointers
 * @param device_in_bytes Pointer to array of compressed sizes
 * @param device_out_bytes Pointer to array where uncompressed sizes will be
 * stored
 * @param num_chunks Number of chunks to process
 */
__global__ void __launch_bounds__(1) get_zstd_uncompressed_sizes_kernel(
    const void *const *__restrict__ device_in_ptr,
    const uint64_t *__restrict__ device_in_bytes,
    uint64_t *__restrict__ device_out_bytes, int num_chunks) {
  int chunk_id = threadIdx.x + blockDim.x * blockIdx.x;
  if (chunk_id < num_chunks) {
    const uint8_t *comp_data =
        reinterpret_cast<const uint8_t *>(device_in_ptr[chunk_id]);
    uint64_t comp_size = device_in_bytes[chunk_id];

    // Use the robust get_uncompressed_size function
    unsigned long long size = zstd::Decompressor::get_uncompressed_size(
        comp_data, comp_size, nullptr);

    // ZSTD error constants
    constexpr unsigned long long ZSTD_CONTENTSIZE_ERROR = 0xFFFFFFFFFFFFFFFEULL;

    // If we got an error, set size to 0 to indicate failure
    if (size == ZSTD_CONTENTSIZE_ERROR) {
      device_out_bytes[chunk_id] = 0;
    } else {
      device_out_bytes[chunk_id] = size;
    }
  }
}

/**
 * @brief Get uncompressed sizes for ZSTD compressed chunks.
 *
 * @param device_in_ptr Pointer to array of compressed data pointers
 * @param device_in_bytes Pointer to array of compressed sizes
 * @param device_out_bytes Pointer to array where uncompressed sizes will be
 * stored
 * @param count Number of chunks
 * @param stream HIP stream to use
 * @throws HipCompException if any chunk has an invalid ZSTD frame (size == 0)
 */
void gpu_zstd_get_uncompressed_sizes(const void *const *device_in_ptr,
                                     const size_t *device_in_bytes,
                                     size_t *device_out_bytes, int count,
                                     hipStream_t stream) {
  // NOTE: Only a single thread is used by this kernel per chunk
  dim3 dim_block(1);
  dim3 dim_grid(count, 1);

  get_zstd_uncompressed_sizes_kernel<<<dim_grid, dim_block, 0, stream>>>(
      device_in_ptr, device_in_bytes, device_out_bytes, count);
  HipUtils::check_last_error(
      "Failed to run ZSTD kernel gpu_zstd_get_uncompressed_sizes");

#if DEBUGLEVEL > 0
  // Validate that all chunks have valid sizes (non-zero)
  // Copy results to host to check for errors
  std::vector<size_t> host_out_bytes(count);
  CHECK_HIP_API_CALL(hipMemcpyAsync(host_out_bytes.data(), device_out_bytes,
                                    count * sizeof(size_t),
                                    hipMemcpyDeviceToHost, stream));
  CHECK_HIP_API_CALL(hipStreamSynchronize(stream));

  // Check for any zero entries indicating invalid ZSTD frames
  for (int i = 0; i < count; ++i) {
    if (host_out_bytes[i] == 0) {
      throw HipCompException(
          hipcompErrorCannotDecompress,
          "ZSTD frame header is invalid or cannot be read for chunk " +
              std::to_string(i));
    }
  }
#endif
}

} // namespace hipcomp
