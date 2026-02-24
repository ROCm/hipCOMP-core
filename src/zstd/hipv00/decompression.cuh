/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file gpudecompress.hip

  Derived from zlib's contrib/puff.c, original copyright notice below

*/

/*
Copyright (C) 2002-2013 Mark Adler, all rights reserved
version 2.3, 21 Jan 2013

This software is provided 'as-is', without any express or implied
warranty.  In no event will the author be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

Mark Adler    madler@alumni.caltech.edu
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

#pragma once

#include "decompression_decode.cuh"
#include "device_functions.cuh"
#include "hip/hip_runtime.h"

#include <hipcub/hipcub.hpp>

namespace hipcomp {
namespace zstd {

/**
 * ZSTD decompressor.
 *
 * \note We may introduce template parameters for the decompressor class in the
 * future.
 */
class Decompressor {
private:
  static DEVICE_INLINE void
  decompress(const uint8_t *const __restrict__ device_in_ptr, // ptr
             const uint64_t device_in_bytes,
             uint8_t *const __restrict__ device_out_ptr,
             const uint64_t device_out_available_bytes,
             hipcompStatus_t *const __restrict__ status,      // &scalar
             uint64_t *const __restrict__ device_out_bytes) { // &scalar
    int t = threadIdx.x;

    if (!t) {
      DEBUGLOG(5, "Decompressor::decompress: enter ZSTD_decompress");

      // Call ZSTD_decompress with 4 arguments
      size_t result = ZSTD_decompress(device_out_ptr,             // dst
                                      device_out_available_bytes, // dstCapacity
                                      device_in_ptr,              // src
                                      device_in_bytes             // srcSize
      );

      // Check if decompression was successful
      // ZSTD_isError() checks if the result is an error code
      if (ZSTD_isError(result)) {
        *status = hipcompStatus_t::hipcompErrorCannotDecompress;
        if (device_out_bytes) {
          *device_out_bytes = 0;
        }
      } else {
        *status = hipcompStatus_t::hipcompSuccess;
        if (device_out_bytes) {
          *device_out_bytes = result; // result is the actual decompressed size
        }
      }
    }
  }

public:
  static constexpr int DECOMP_WARPS_PER_BLOCK =
      1; // single warp & thread per threadblock

  static DEVICE_INLINE void
  apply(const uint8_t *const __restrict__ device_in_ptr, // ptr
        const uint64_t device_in_bytes,
        uint8_t *const __restrict__ device_out_ptr,
        const uint64_t device_out_available_bytes,
        hipcompStatus_t *const __restrict__ status, // &scalar
        uint64_t *const __restrict__ device_out_bytes) {
    decompress(device_in_ptr, device_in_bytes, device_out_ptr,
               device_out_available_bytes, status, device_out_bytes);
  }

  /**
   * @brief Get the actual uncompressed size from a ZSTD frame.
   *
   * This reads the frame header to determine the exact uncompressed size.
   *
   * @param device_in_ptr Pointer to compressed ZSTD data
   * @param device_in_bytes Size of compressed data
   * @return The uncompressed size if known, or:
   *         - ZSTD_CONTENTSIZE_UNKNOWN (0xFFFFFFFFFFFFFFFFULL) if size cannot
   * be determined
   *         - ZSTD_CONTENTSIZE_ERROR (0xFFFFFFFFFFFFFFFEULL) if an error
   * occurred
   */
  static DEVICE_INLINE unsigned long long
  get_frame_content_size(const uint8_t *const __restrict__ device_in_ptr,
                         const uint64_t device_in_bytes) {
    return ZSTD_getFrameContentSize(device_in_ptr, device_in_bytes);
  }

  /**
   * @brief Get the total decompressed size for all frames in the input.
   *
   * This function iterates through all ZSTD frames to calculate total size.
   *
   * @param device_in_ptr Pointer to compressed ZSTD data (may contain multiple
   * frames)
   * @param device_in_bytes Size of compressed data
   * @return The total uncompressed size of all frames, or:
   *         - ZSTD_CONTENTSIZE_ERROR (0xFFFFFFFFFFFFFFFEULL) if an error
   * occurred
   */
  static DEVICE_INLINE unsigned long long
  get_decompressed_size(const uint8_t *const __restrict__ device_in_ptr,
                        const uint64_t device_in_bytes) {
    return ZSTD_findDecompressedSize(device_in_ptr, device_in_bytes);
  }

  /**
   * @brief Get the maximum possible decompressed size (bound).
   *
   * This provides an upper bound for the decompressed size, which is useful
   * for allocating output buffers when the exact size is unknown.
   *
   * @param device_in_ptr Pointer to compressed ZSTD data
   * @param device_in_bytes Size of compressed data
   * @return The maximum decompressed size, or ZSTD_CONTENTSIZE_ERROR on error
   */
  static DEVICE_INLINE unsigned long long
  get_decompressed_bound(const uint8_t *const __restrict__ device_in_ptr,
                         const uint64_t device_in_bytes) {
    return ZSTD_decompressBound(device_in_ptr, device_in_bytes);
  }

  /**
   * @brief Get the most accurate output buffer size estimate in a robust way.
   *
   * This function tries multiple methods in order of accuracy:
   * 1. First tries get_decompressed_size() for exact size across all frames
   * 2. Falls back to get_frame_content_size() for single frame exact size
   * 3. Finally uses get_decompressed_bound() for upper bound estimate
   *
   * @param device_in_ptr Pointer to compressed ZSTD data
   * @param device_in_bytes Size of compressed data
   * @param is_exact Output parameter - set to true if exact size was found,
   * false if estimate
   * @return The output buffer size (exact or upper bound), or
   * ZSTD_CONTENTSIZE_ERROR on error
   */
  static DEVICE_INLINE unsigned long long
  get_uncompressed_size(const uint8_t *const __restrict__ device_in_ptr,
                        const uint64_t device_in_bytes,
                        bool *is_exact = nullptr) {
    // Try 1: get_decompressed_size() - most accurate, handles multiple frames
    unsigned long long size =
        get_decompressed_size(device_in_ptr, device_in_bytes);
    if (size != ZSTD_CONTENTSIZE_ERROR && size != ZSTD_CONTENTSIZE_UNKNOWN) {
      if (is_exact)
        *is_exact = true;
      return size;
    }

    // Try 2: get_frame_content_size() - exact for single frame
    size = get_frame_content_size(device_in_ptr, device_in_bytes);
    if (size != ZSTD_CONTENTSIZE_ERROR && size != ZSTD_CONTENTSIZE_UNKNOWN) {
      if (is_exact)
        *is_exact = true;
      return size;
    }

    // Try 3: get_decompressed_bound() - fallback to upper bound
    size = get_decompressed_bound(device_in_ptr, device_in_bytes);
    if (size != ZSTD_CONTENTSIZE_ERROR) {
      if (is_exact)
        *is_exact = false;
      return size;
    }

    // All methods failed
    if (is_exact)
      *is_exact = false;
    return ZSTD_CONTENTSIZE_ERROR;
  }
};

/**
 * \brief Per-chunk (=single block) operation of ZSTD decompression
 *
 * Inputs are not arrays of device buffers but single device buffers.
 *
 * The address of scalar arguments ``status`` and ``device_out_bytes``
 * is passed to the device function so that thread 0 can write back the
 * respective values.
 *
 * @brief Interface for decompressing ZSTD-compressed data
 *
 * Multiple, independent chunks of compressed data can be decompressed by using
 * separate input/output/status for each chunk.
 *
 * @param[in] device_in_ptr Source buffer
 * @param[in] device_in_bytes Source buffer size
 * @param[in] device_out_ptr Destination buffer
 * @param[in] device_out_available_bytes Destination buffer size
 * @param[out] status Decompression status
 * @param[out] device_out_bytes Actual decompressed size
 */
__device__ void
do_decompress2(const uint8_t *const __restrict__ device_in_ptr, // ptr
               const uint64_t device_in_bytes,
               uint8_t *const __restrict__ device_out_ptr,
               const uint64_t device_out_available_bytes,
               hipcompStatus_t *const __restrict__ status,      // &scalar
               uint64_t *const __restrict__ device_out_bytes) { // &scalar
  Decompressor::apply(device_in_ptr, device_in_bytes, device_out_ptr,
                      device_out_available_bytes, status, device_out_bytes);
}

/**
 * \note This is an alias for `do_decompress2`.
 *
 */
__device__ void
do_decompress(const uint8_t *const __restrict__ device_in_ptr, // ptr
              const uint64_t device_in_bytes,
              uint8_t *const __restrict__ device_out_ptr,
              const uint64_t device_out_available_bytes,
              hipcompStatus_t *const __restrict__ status, // &scalar
              uint64_t *const __restrict__ device_out_bytes) {
  do_decompress2(device_in_ptr, device_in_bytes, device_out_ptr,
                 device_out_available_bytes, status, device_out_bytes);
}

constexpr int DECOMP_WARPS_PER_BLOCK = Decompressor::DECOMP_WARPS_PER_BLOCK;

} // namespace zstd
} // namespace hipcomp
