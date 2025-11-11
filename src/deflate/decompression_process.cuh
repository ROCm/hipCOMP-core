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

#include "device_functions.cuh"

namespace hipcomp {
namespace deflate {

/// @brief WARP1: process LZ77 symbols and output uncompressed stream
template <int warpsize, typename INFLATE_STATE_S> class ProcessSymbols {
private:
  using MaskT = typename Mask<warpsize>::type;
  static constexpr int BATCH_SIZE = INFLATE_STATE_S::BATCH_SIZE;
  static constexpr int BATCH_COUNT = INFLATE_STATE_S::BATCH_COUNT;
  static constexpr int PREFETCH_SIZE = INFLATE_STATE_S::PREFETCH_SIZE;

public:
  /**
   *  \brief Applies the strategy.
   *
   * \param[inout] s decompression state
   * \param[ino] t thread id within participating group (lane id)
   */
  __device__ static inline void apply(INFLATE_STATE_S *s, int t) {
    uint8_t *out = s->out;
    uint8_t const *outend = s->outend;
    uint8_t const *outbase = s->outbase;
    int batch = 0;

    do {
      volatile uint32_t *b = &s->x.u.symqueue[batch * BATCH_SIZE];
      int batch_len = 0;
      if (t == 0) {
        while ((batch_len = s->x.batch_len[batch]) == 0) {
        }
      }
      batch_len = SHFL10(batch_len);
      if (batch_len < 0) {
        break;
      }

      auto const symt = (t < batch_len) ? b[t] : 256;
      const MaskT lit_mask = BALLOT1<MaskT>(symt >= 256);
      auto pos = static_cast<uint8_t>(
          min((find_first_set_bit(lit_mask) - 1) & 0xff,
              warpsize)); // TODO(HIP/AMD): static_cast safe here (WAR for
                          // compiler error)?, also: check mask, warp size and
                          // __ffs invocation

      if (t == 0) {
        s->x.batch_len[batch] = 0;
      }

      if (t < pos && out + t < outend) {
        out[t] = symt;
      }
      out += pos;
      batch_len -= pos;
      while (batch_len > 0) {
        int dist, len, symbol;

        // Process a non-literal symbol
        symbol = SHFL1(symt, pos);
        len = max((symbol & 0xffff) - 256,
                  0); // max should be unnecessary, but just in case
        dist = symbol >> 16;
        for (int i = t; i < len; i += 32) {
          uint8_t const *src = out + ((i >= dist) ? (i % dist) : i) - dist;
          if (out + i < outend and src >= outbase) {
            out[i] = *src;
          }
        }
        out += len;
        pos++;
        batch_len--;
        // Process subsequent literals, if any
        if (!((lit_mask >> pos) & 1)) {
          len =
              min((find_first_set_bit(lit_mask >> pos) - 1) & 0xff, batch_len);
          symbol = SHFL1(symt, (pos + t) & (warpsize - 1));
          if (t < len && out + t < outend) {
            out[t] = symbol;
          }
          out += len;
          pos += len;
          batch_len -= len;
        }
      }
      batch = (batch + 1) & (BATCH_COUNT - 1);
    } while (true);

    if (t == 0) {
      s->out = out;
    }
  }
};

} // namespace deflate
} // namespace hipcomp
