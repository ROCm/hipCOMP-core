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

// NOTE: Currently unused!
namespace hipcomp {
namespace gzip {

/**
 * @brief Copy a group of buffers
 *
 * blockDim {1024,1,1}
 *
 * @param inputs Source and destination information per block
 */
__global__ void __launch_bounds__(1024) copy_uncompressed_kernel(
    device_span<device_span<uint8_t const> const> inputs,
    device_span<device_span<uint8_t> const> outputs) {
  __shared__ uint8_t const *volatile src_g;
  __shared__ uint8_t *volatile dst_g;
  __shared__ uint32_t volatile copy_len_g;

  uint32_t t = threadIdx.x;
  uint32_t z = blockIdx.x;
  uint8_t const *src;
  uint8_t *dst;
  uint32_t len, src_align_bytes, src_align_bits, dst_align_bytes;

  if (!t) {
    src = inputs[z].data();
    dst = outputs[z].data();
    len = static_cast<uint32_t>(min(inputs[z].size(), outputs[z].size()));
    src_g = src;
    dst_g = dst;
    copy_len_g = len;
  }
  __syncthreads();
  src = src_g;
  dst = dst_g;
  len = copy_len_g;
  // Align output to 32-bit
  dst_align_bytes = 3 & -reinterpret_cast<intptr_t>(dst);
  if (dst_align_bytes != 0) {
    uint32_t align_len = min(dst_align_bytes, len);
    if (t < align_len) {
      dst[t] = src[t];
    }
    src += align_len;
    dst += align_len;
    len -= align_len;
  }
  src_align_bytes = (uint32_t)(3 & reinterpret_cast<uintptr_t>(src));
  src_align_bits = src_align_bytes << 3;
  while (len >= 32) {
    auto const *src32 =
        reinterpret_cast<uint32_t const *>(src - src_align_bytes);
    uint32_t copy_cnt = min(len >> 2, 1024);
    if (t < copy_cnt) {
      uint32_t v = src32[t];
      if (src_align_bits != 0) {
        v = __funnelshift_r(v, src32[t + 1], src_align_bits);
      }
      reinterpret_cast<uint32_t *>(dst)[t] = v;
    }
    src += copy_cnt * 4;
    dst += copy_cnt * 4;
    len -= copy_cnt * 4;
  }
  if (t < len) {
    dst[t] = src[t];
  }
}

void gpu_copy_uncompressed_blocks(
    device_span<device_span<uint8_t const> const> inputs,
    device_span<device_span<uint8_t> const> outputs, hipStream_t stream) {
  if (inputs.size() > 0) {
    copy_uncompressed_kernel<<<inputs.size(), 1024, 0, stream>>>(inputs,
                                                                 outputs);
  }
}

} // namespace gzip
} // namespace hipcomp
