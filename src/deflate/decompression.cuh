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

/** @file gpuinflate.hip

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
// Modifications Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights
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
#include "decompression_prefetch.cuh"
#include "decompression_process.cuh"
#include "device_functions.cuh"
#include "hip/hip_runtime.h"

#include <hipcub/hipcub.hpp>

namespace hipcomp {
namespace deflate {

/**
 * @brief GZIP header flags
 * See https://tools.ietf.org/html/rfc1952
 */
namespace GZIPHeaderFlag {
constexpr uint8_t ftext = 0x01;    // ASCII text hint
constexpr uint8_t fhcrc = 0x02;    // Header CRC present
constexpr uint8_t fextra = 0x04;   // Extra fields present
constexpr uint8_t fname = 0x08;    // Original file name present
constexpr uint8_t fcomment = 0x10; // Comment present
}; // namespace GZIPHeaderFlag

/**
 * @brief Parse GZIP header
 * See https://tools.ietf.org/html/rfc1952
 */
__device__ int parse_gzip_header(uint8_t const *src, size_t src_size) {
  int hdr_len = -1;

  if (src_size >= 18) {
    uint32_t sig = (src[0] << 16) | (src[1] << 8) | src[2];
    if (sig == 0x1f'8b08) // 24-bit GZIP inflate signature {0x1f, 0x8b, 0x08}
    {
      uint8_t flags = src[3];
      hdr_len = 10;
      if (flags & GZIPHeaderFlag::fextra) // Extra fields present
      {
        int xlen = src[hdr_len] | (src[hdr_len + 1] << 8);
        hdr_len += xlen;
        if (hdr_len >= src_size)
          return -1;
      }
      if (flags & GZIPHeaderFlag::fname) // Original file name present
      {
        // Skip zero-terminated string
        do {
          if (hdr_len >= src_size)
            return -1;
        } while (src[hdr_len++] != 0);
      }
      if (flags & GZIPHeaderFlag::fcomment) // Comment present
      {
        // Skip zero-terminated string
        do {
          if (hdr_len >= src_size)
            return -1;
        } while (src[hdr_len++] != 0);
      }
      if (flags & GZIPHeaderFlag::fhcrc) // Header CRC present
      {
        hdr_len += 2;
      }
      if (hdr_len + 8 >= src_size)
        hdr_len = -1;
    }
  }
  return hdr_len;
}

/**
 * Deflate/GZIP compressor.
 *
 * \tparam PREFETCH_SIZE Number of bytes to prefetch, only use if temmplate
 *                          parameter \code ENABLE_PREFETCH \endcode is set to
 *                          \code true \endcode, can be set to anything (0)
 * otherwise.
 * \tparam ENABLE_PREFETCH Enable the use of a specialized prefetch warp.
 */
template <int warpsize, int BATCH_SIZE, int BATCH_COUNT, int PREFETCH_SIZE,
          bool ENABLE_PREFETCH>
class Decompressor {
private:
  using INFLATE_STATE_S =
      inflate_state_s<BATCH_SIZE, BATCH_COUNT, PREFETCH_SIZE>;

  template <typename DECODER, typename PREFETCHER, typename PROCESSOR>
  static __device__ inline void
  decompress(const uint8_t *const __restrict__ device_in_ptr, // ptr
             const uint64_t device_in_bytes,
             uint8_t *const __restrict__ device_out_ptr,
             const uint64_t device_out_available_bytes,
             hipcompStatus_t *const __restrict__ status,    // &scalar
             uint64_t *const __restrict__ device_out_bytes, // &scalar
             uint32_t *const __restrict__ device_reserved,  // &scalar
             bool parse_hdr) {
    __shared__ __align__(16) INFLATE_STATE_S state_g;

    int t = threadIdx.x;
    //: - int z                  = blockIdx.x;
    INFLATE_STATE_S *state = &state_g;

    if (!t) {
      //:- auto p        = inputs[z].data();
      //:- auto src_size = inputs[z].size();
      auto p = device_in_ptr;
      auto src_size = device_in_bytes;
      // Parse header if needed
      state->err = 0;
      if (parse_hdr) {
        int hdr_len = parse_gzip_header(p, src_size);
        src_size = (src_size >= 8) ? src_size - 8 : 0; // ignore footer
        if (hdr_len >= 0) {
          p += hdr_len;
          src_size -= hdr_len;
        } else {
          state->err = hdr_len;
        }
      }
      // Initialize shared state
      //:- state->out              = outputs[z].data();
      state->out = device_out_ptr;
      state->outbase = state->out;
      //:- state->outend           = state->out + outputs[z].size();
      state->outend = state->out + device_out_available_bytes;
      state->end = p + src_size;
      auto const prefix_bytes = (uint32_t)(((size_t)p) & 3);
      p -= prefix_bytes;
      state->cur = p;
      state->bitbuf.x =
          (p < state->end) ? *reinterpret_cast<uint32_t const *>(p) : 0;
      p += 4;
      state->bitbuf.y =
          (p < state->end) ? *reinterpret_cast<uint32_t const *>(p) : 0;
      state->bitpos = prefix_bytes * 8;
    }
    __syncthreads();
    // Main loop decoding blocks
    while (!state->err) {
      if (!t) {
        // Thread0: read last flag, block type and custom huffman tables if any
        if (state->cur + (state->bitpos >> 3) >= state->end)
          state->err = 2;
        else {
          state->blast = state->getbits(1);
          state->btype = state->getbits(2);
          if (state->btype == 0)
            state->err = state->init_stored();
          else if (state->btype == 1)
            state->err = state->init_fixed();
          else if (state->btype == 2)
            state->err = state->init_dynamic();
          else
            state->err = -1; // Invalid block
        }
      }
      __syncthreads();
      if (!state->err && (state->btype == 1 || state->btype == 2)) {
        // Initializes lookup tables (block wide)
        state->init_length_lut(t);
        state->init_distance_lut(t);
        if constexpr (ENABLE_PREFETCH) {
          // Initialize prefetcher
          state->init_prefetcher(t);
        }
        if (t < BATCH_COUNT) {
          state->x.batch_len[t] = 0;
        }
        __syncthreads();
        // decode data until end-of-block code
        if (t < 1 * warpsize) {
          // WARP0: decode variable-length symbols
          DECODER::apply(state, t);
          if (!t) {
            if constexpr (ENABLE_PREFETCH) {
              state->pref.run = 0;
            }
          }
        } else if (t < 2 * warpsize) {
          // WARP1: perform LZ77 using length and distance codes from WARP0
          PROCESSOR::apply(state, t & (warpsize - 1));
        } else if (t < 3 * warpsize) {
          if constexpr (ENABLE_PREFETCH) {
            // WARP2: Prefetcher: prefetch data for WARP0
            PREFETCHER::apply(state, t & (warpsize - 1));
          }
        }
        // else WARP3: idle
      } else if (!state->err && state->btype == 0) {
        // Uncompressed block (block-wide memcpy)
        state->copy_stored(t);
      }
      if (state->blast)
        break;
      __syncthreads();
    }
    __syncthreads();
    // Output decompression status and length
    if (!t) {
      if (state->err == 0 &&
          state->cur + ((state->bitpos + 7) >> 3) > state->end) {
        // Read past the end of the input buffer
        state->err = 2;
      } else if (state->err == 0 && state->out > state->outend) {
        // Output buffer too small
        state->err = 1;
      }
      //:- results[z].bytes_written = state->out - state->outbase;
      //:- results[z].status        = [&]() {
      //:- switch (state->err) {
      //:-   case 0: return compression_status::SUCCESS;
      //:-   case 1: return compression_status::OUTPUT_OVERFLOW;
      //:-   default: return compression_status::FAILURE;
      //:- }
      //:- }();
      //:- results[z].reserved = (int)(state->end - state->cur);  // Here mainly
      //: for debug purposes
      if (device_out_bytes) {
        *device_out_bytes = state->out - state->outbase;
      }
      if (status) {
        *status = [&]() {
          switch (state->err) {
          case 0:
            return hipcompSuccess;
          case 1:
            return hipcompErrorOutOfMemory;
          default:
            return hipcompErrorCannotDecompress;
          }
        }();
      }
      if (device_reserved) {
        *device_reserved =
            (int)(state->end - state->cur); // Here mainly for debug purposes
      }
    }
  }

public:
  // TODO(HIP/AMD): Double-check meaning of launch_bounds HIP vs CUDA
  // TODO(HIP/AMD): we have 3 (2 without prefetcher) specialized warps, why 4
  // warps per block?
  // TODO(HIP/AMD): make this constexpr w
  static constexpr int DECOMP_WARPS_PER_BLOCK = 4;

  static __device__ inline void
  apply(const uint8_t *const __restrict__ device_in_ptr, // ptr
        const uint64_t device_in_bytes,
        uint8_t *const __restrict__ device_out_ptr,
        const uint64_t device_out_available_bytes,
        hipcompStatus_t *const __restrict__ status,    // &scalar
        uint64_t *const __restrict__ device_out_bytes, // &scalar
        uint32_t *const __restrict__ device_reserved,  // &scalar
        bool parse_hdr) {
    decompress<
        DecodeSymbolsSingleThreaded<warpsize, INFLATE_STATE_S, ENABLE_PREFETCH>,
        PrefetchByteStream<warpsize, INFLATE_STATE_S>,
        ProcessSymbols<warpsize,
                       INFLATE_STATE_S>>(device_in_ptr, // ptr
                                         device_in_bytes, device_out_ptr,
                                         device_out_available_bytes,
                                         status,           // &scalar
                                         device_out_bytes, // &scalar
                                         device_reserved,  // &scalar
                                         parse_hdr);
  }
};

/**
 * \brief Per-chunk (=single block) operation of INFLATE ("DEFLATE
 * decompression").
 *
 * Inputs are not arrays of device buffers but single device buffers.
 *
 * The address of scalar arguments ``status``, ``device_out_bytes``, and
 * ``device_reserved`` is passed to the device function so that thread 0
 * can write back the respective values.
 *
 * @brief Interface for decompressing GZIP-compressed data
 *
 * Multiple, independent chunks of compressed data can be decompressed by using
 * separate input/output/status for each chunk.
 *
 * @param[in] device_in_ptr Source buffer
 * @param[in] device_in_bytes Source buffer size
 * @param[in] device_out_ptr Destination buffer
 * @param[in] device_out_bytes Destination buffer size
 * @param[out] status Decompression status.
 * @param[out] device_written_bytes Decompression status.
 * @param[out] device_reserved An integer for additional information, i.e., for
 * debugging.
 * @param[in] parse_hdr If ``true``, indicates that the compressed bitstream
 * includes a GZIP header
 */
template <int warpsize, int BATCH_SIZE, int BATCH_COUNT, int PREFETCH_SIZE,
          bool ENABLE_PREFETCH = true>
__device__ void
do_inflate2(const uint8_t *const __restrict__ device_in_ptr, // ptr
            const uint64_t device_in_bytes,
            uint8_t *const __restrict__ device_out_ptr,
            const uint64_t device_out_available_bytes,
            hipcompStatus_t *const __restrict__ status,    // &scalar
            uint64_t *const __restrict__ device_out_bytes, // &scalar
            uint32_t *const __restrict__ device_reserved,  // &scalar
            bool parse_hdr) {
  Decompressor<warpsize, BATCH_SIZE, BATCH_COUNT, PREFETCH_SIZE,
               ENABLE_PREFETCH>::apply(device_in_ptr, // ptr
                                       device_in_bytes, device_out_ptr,
                                       device_out_available_bytes,
                                       status,           // &scalar
                                       device_out_bytes, // &scalar
                                       device_reserved,  // &scalar
                                       parse_hdr);
}

/**
 * \note This variant of "do_inflate" pre-defines some of the template
 *       parmeters of `do_inflate2`.
 */
template <int warpsize>
__device__ void
do_inflate(const uint8_t *const __restrict__ device_in_ptr, // ptr
           const uint64_t device_in_bytes,
           uint8_t *const __restrict__ device_out_ptr,
           const uint64_t device_out_available_bytes,
           hipcompStatus_t *const __restrict__ status,    // &scalar
           uint64_t *const __restrict__ device_out_bytes, // &scalar
           uint32_t *const __restrict__ device_reserved,  // &scalar
           bool parse_hdr) {
  // TODO(HIP/AMD): Fine-tune these parameters.
  // We need a prefetch size of >=2^10 = 1024 bytes here.
  // At batch_size = 2^6 = 64 for HIP AMD backend, wavefront 0
  // requires the prefetcher (wavefront 2) to have prefetched at least 64 * 6 =
  // 384 bytes before proceeding with process_symbols(). However, with
  // prefetch_size<=2^9 = 512 bytes, the prefetcher would only fetch 256 bytes
  // (some space is left on purpose in the buffer), which would cause some unit
  // tests to hang.
  constexpr int log2_prefetch_size =
      (warpsize == 64) ? 10 : 9; // Must be at least LOG2_BATCH_SIZE+3
  constexpr int prefetch_size = (1 << log2_prefetch_size);

  constexpr int log2_batch_size = (warpsize == 64) ? 6 : 5;
  constexpr int batch_size = (1 << log2_batch_size);

  /// 4 batches of 32 symbols
  constexpr int log2_batch_count = 2; // 1..5
  constexpr int batch_count = (1 << log2_batch_count);

  constexpr bool enable_prefetch = true;

  do_inflate2<warpsize, batch_size, batch_count, prefetch_size,
              enable_prefetch>(device_in_ptr, // ptr
                               device_in_bytes, device_out_ptr,
                               device_out_available_bytes,
                               status,           // &scalar
                               device_out_bytes, // &scalar
                               device_reserved,  // &scalar
                               parse_hdr);
}

constexpr int DECOMP_WARPS_PER_BLOCK =
    Decompressor<0, 0, 0, 0, true>::DECOMP_WARPS_PER_BLOCK;

} // namespace deflate
} // namespace hipcomp

// NOTE: functions in `deflate/helpers.hiph` currently unused
// #include "deflate/helpers.cuh"
