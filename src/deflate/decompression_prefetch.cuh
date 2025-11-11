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

/**
 * Prefetch byte stream strategy that needs
 * to be passed ot the Decompressor class.
 */
template <int warpsize, typename INFLATE_STATE_S> class PrefetchByteStream {
private:
  static constexpr int PREFETCH_SIZE = INFLATE_STATE_S::PREFETCH_SIZE;

public:
  /**
   * \brief Applies the strategy.
   *
   * \param[inout] s decompression state
   * \param[in] t warp lane index, i.e. threadIdx.x % warpsize.
   */
  __device__ static inline void apply(volatile INFLATE_STATE_S *s,
                                      const int t) {
    uint8_t const *cur_p = s->pref.cur_p;
    uint8_t const *end = s->end;
    while (SHFL10((t == 0) ? s->pref.run : 0)) {
      auto cur_lo = (int32_t)(size_t)cur_p;
      int do_pref = SHFL10((t == 0) ? (cur_lo - *(volatile int32_t *)&s->cur <
                                       PREFETCH_SIZE - warpsize * 4 - 4)
                                    : 0);
      if (do_pref) {
        uint8_t const *p = cur_p + 4 * t;
        *prefetch_addr32(s->pref, p) =
            (p < end) ? *reinterpret_cast<uint32_t const *>(p) : 0;
        cur_p += 4 * warpsize;
        __threadfence_block();
        SYNCWARP();
        if (!t) {
          s->pref.cur_p = cur_p;
          __threadfence_block();
        }
      }
    }
  }
};

} // namespace deflate
} // namespace hipcomp
