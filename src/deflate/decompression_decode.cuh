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

#include "cstdint"
#include "decompression_state.cuh"
#include "device_functions.cuh"
#include "hip/hip_runtime.h"
#include "hipcub/hipcub.hpp"

namespace hipcomp {
namespace deflate {

__device__ inline unsigned int bfe(unsigned int source, unsigned int bit_start,
                                   unsigned int num_bits) {
  // TODO(HIP/AMD): check if we have an equivalent ISA instruction/intrinsic
  // that can be used here unsigned int bits; asm("bfe.u32 %0, %1, %2, %3;" :
  // "=r"(bits) : "r"(source), "r"(bit_start), "r"(num_bits));
  return hipcub::BFE(source, bit_start, num_bits);
}

/**
 * \brief Decode symbols and output LZ77 batches (single-warp).
 */
template <int warpsize, typename INFLATE_STATE_S, bool ENABLE_PREFETCH>
class DecodeSymbolsSingleThreaded {
private:
  static constexpr int BATCH_SIZE = INFLATE_STATE_S::BATCH_SIZE;
  static constexpr int BATCH_COUNT = INFLATE_STATE_S::BATCH_COUNT;

  /**
   * @brief Decode literal/length and distance codes until an end-of-block code.
   *
   * Format notes:
   *
   * - Compressed data that is after the block type if fixed or after the code
   *   description if dynamic is a combination of literals and length/distance
   *   pairs terminated by and end-of-block code.  Literals are simply Huffman
   *   coded bytes.  A length/distance pair is a coded length followed by a
   *   coded distance to represent a string that occurs earlier in the
   *   uncompressed data that occurs again at the current location.
   *
   * - Literals, lengths, and the end-of-block code are combined into a single
   *   code of up to 286 symbols.  They are 256 literals (0..255), 29 length
   *   symbols (257..285), and the end-of-block symbol (256).
   *
   * - There are 256 possible lengths (3..258), and so 29 symbols are not enough
   *   to represent all of those.  Lengths 3..10 and 258 are in fact represented
   *   by just a length symbol.  Lengths 11..257 are represented as a symbol and
   *   some number of extra bits that are added as an integer to the base length
   *   of the length symbol.  The number of extra bits is determined by the base
   *   length symbol.  These are in the static arrays below, lens[] for the base
   *   lengths and lext[] for the corresponding number of extra bits.
   *
   * - The reason that 258 gets its own symbol is that the longest length is
   * used often in highly redundant files.  Note that 258 can also be coded as
   * the base value 227 plus the maximum extra value of 31.  While a good
   * deflate should never do this, it is not an error, and should be decoded
   * properly.
   *
   * - If a length is decoded, including its extra bits if any, then it is
   *   followed a distance code.  There are up to 30 distance symbols.  Again
   *   there are many more possible distances (1..32768), so extra bits are
   * added to a base value represented by the symbol.  The distances 1..4 get
   * their own symbol, but the rest require extra bits.  The base distances and
   *   corresponding number of extra bits are below in the static arrays dist[]
   *   and dext[].
   *
   * - Literal bytes are simply written to the output.  A length/distance pair
   * is an instruction to copy previously uncompressed bytes to the output.  The
   *   copy is from distance bytes back in the output stream, copying for length
   *   bytes.
   *
   * - Distances pointing before the beginning of the output data are not
   *   permitted.
   *
   * - Overlapped copies, where the length is greater than the distance, are
   *   allowed and common.  For example, a distance of one and a length of 258
   *   simply copies the last byte 258 times.  A distance of four and a length
   * of twelve copies the last four bytes three times.  A simple forward copy
   *   ignoring whether the length is greater than the distance or not
   * implements this correctly.  You should not use memcpy() since its behavior
   * is not defined for overlapped arrays.  You should not use memmove() or
   * bcopy() since though their behavior -is- defined for overlapping arrays, it
   * is defined to do the wrong thing in this case.
   */

public:
  /// @brief Thread 0 only: decode bitstreams and output symbols into the symbol
  /// queue
  __device__ static inline void apply(INFLATE_STATE_S *s, int t = 0) {
    uint32_t bitpos = s->bitpos;
    uint2 bitbuf = s->bitbuf;
    auto cur = s->cur;
    auto end = s->end;
    int32_t batch = 0;
    int32_t sym, batch_len;

    if (t > 0) {
      return;
    }

    // Thread0: decode symbols (single threaded)
    do {
      volatile uint32_t *b = &s->x.u.symqueue[batch * BATCH_SIZE];
      // Wait for the next batch entry to be empty
      if constexpr (ENABLE_PREFETCH) {
        // Wait for prefetcher to fetch a worst-case of 48 bits per symbol
        while ((*(volatile int32_t *)&s->pref.cur_p - (int32_t)(size_t)cur <
                BATCH_SIZE * 6) ||
               (s->x.batch_len[batch] != 0)) {
        }
      } else {
        while (s->x.batch_len[batch] != 0) {
        }
      }
      batch_len = 0;
      if constexpr (ENABLE_PREFETCH) {
        if (cur + (bitpos >> 3) >= end) {
          s->err = 1;
          break;
        }
      }
      // Inner loop decoding symbols
      do {
        uint32_t next32 =
            __funnelshift_rc(bitbuf.x, bitbuf.y, bitpos); // nextbits32(s);
        uint32_t len;
        sym = s->u.lut.lenlut[next32 & ((1 << log2_len_lut) - 1)];
        if ((uint32_t)sym < (uint32_t)(0x100 << 5)) {
          // We can lookup a second symbol if this was a short literal
          len = sym & 0x1f;
          sym >>= 5;
          b[batch_len++] = sym;
          next32 >>= len;
          bitpos += len;
          sym = s->u.lut.lenlut[next32 & ((1 << log2_len_lut) - 1)];
        }
        if (sym > 0) // short symbol
        {
          len = sym & 0x1f;
          sym = ((sym >> 5) & 0x3ff) +
                ((next32 >> (sym >> 24)) & ((sym >> 16) & 0x1f));
        } else {
          // Slow length path
          uint32_t next32r = __brev(next32);
          int16_t const *symbols = &s->lensym[s->index_slow_len];
          unsigned int first = s->first_slow_len;
          int lext;
#pragma unroll 1
          for (len = log2_len_lut + 1; len <= max_bits; len++) {
            unsigned int code = (next32r >> (32 - len)) - first;
            unsigned int count = s->lencnt[len];
            if (code < count) // if length len, return symbol
            {
              sym = symbols[code];
              break;
            }
            symbols += count; // else update for next length
            first += count;
            first <<= 1;
          }
          if (len > max_bits) {
            s->err = -10;
            sym = 256;
            len = 0;
          }
          if (sym > 256) {
            sym -= 257;
            lext = g_lext[sym];
            sym = 256 + g_lens[sym] + bfe(next32, len, lext);
            len += lext;
          }
        }
        if (sym > 256) {
          int dist, dext;
          // skipbits(s, len) inlined - no limit check
          bitpos += len;
          if (bitpos >= 32) {
            bitbuf.x = bitbuf.y;
            if constexpr (ENABLE_PREFETCH) {
              bitbuf.y = *prefetch_addr32(s->pref, cur + 8);
              cur += 4;
            } else {
              cur += 8;
              bitbuf.y = (cur < end) ? *(uint32_t const *)cur : 0;
              cur -= 4;
            }
            bitpos &= 0x1f;
          }
          // get distance
          next32 =
              __funnelshift_rc(bitbuf.x, bitbuf.y, bitpos); // nextbits32(s);
          dist = s->u.lut.distlut[next32 & ((1 << log2_dist_lut) - 1)];
          if (dist > 0) {
            len = dist & 0x1f;
            dext = bfe(dist, 20, 5);
            dist = bfe(dist, 5, 15);
            sym |= (dist + bfe(next32, len, dext)) << 16;
            len += dext;
          } else {
            uint32_t next32r = __brev(next32);
            int16_t const *symbols = &s->distsym[s->index_slow_dist];
            unsigned int first = s->first_slow_dist;
#pragma unroll 1
            for (len = log2_dist_lut + 1; len <= max_bits; len++) {
              unsigned int code = (next32r >> (32 - len)) - first;
              unsigned int count = s->distcnt[len];
              if (code < count) // if length len, return symbol
              {
                dist = symbols[code];
                break;
              }
              symbols += count; // else update for next length
              first += count;
              first <<= 1;
            }
            if (len > max_bits) {
              s->err = -10;
              sym = 256;
              len = 0;
            } else {
              dext = g_dext[dist];
              sym |= (g_dists[dist] + bfe(next32, len, dext)) << 16;
              len += dext;
            }
          }
        }
        // skipbits(s, len) inlined with added error check for reading past
        // the end of the input buffer
        bitpos += len;
        if (bitpos >= 32) {
          bitbuf.x = bitbuf.y;
          if constexpr (ENABLE_PREFETCH) {
            bitbuf.y = *prefetch_addr32(s->pref, cur + 8);
            cur += 4;
          } else {
            cur += 8;
            if (cur < end) {
              bitbuf.y = *(uint32_t const *)cur;
              cur -= 4;
            } else {
              bitbuf.y = 0;
              cur -= 4;
              if (cur > end) {
                s->err = 1;
                sym = 256;
              }
            }
          } // ENABLE_PREFETCH
          bitpos &= 0x1f;
        }
        if (sym == 256)
          break;
        b[batch_len++] = sym;
      } while (batch_len < BATCH_SIZE - 1);
      s->x.batch_len[batch] = batch_len;
      if constexpr (ENABLE_PREFETCH) {
        ((volatile INFLATE_STATE_S *)s)->cur = cur;
      }
      if (batch_len != 0)
        batch = (batch + 1) & (BATCH_COUNT - 1);
    } while (sym != 256);

    while (s->x.batch_len[batch] != 0) {
    }
    s->x.batch_len[batch] = -1;
    s->bitbuf = bitbuf;
    s->bitpos = bitpos;
    if constexpr (!ENABLE_PREFETCH) {
      s->cur = cur;
    }
  }
};

} // namespace deflate
} // namespace hipcomp
