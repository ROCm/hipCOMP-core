/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <hip/hip_runtime.h>

namespace hipcomp {

typedef unsigned Mask32;
typedef unsigned long long Mask64;

template <int warpsize> class Mask {
public:
  using type = typename std::conditional<warpsize == 32, Mask32, Mask64>::type;
};

/**
 * \return the number of consecutive bits of highest significance that contain
 * zeros.
 * \note Return value type matches that of the underlying device builtin.
 *       Changing to unsigned type may result in underflow issues in the
 * dependent code.
 */
template <typename T> __device__ inline int num_leading_zero_bits(T v);

template <> __device__ inline int num_leading_zero_bits<Mask32>(Mask32 v) {
  return __clz(v);
}

template <> __device__ inline int num_leading_zero_bits<Mask64>(Mask64 v) {
  return __clzll(v);
}

/**
 * \return index of first set bit of lowest significance.
 * \note Return value type matches that of the underlying device builtin.
 *       Changing to unsigned type may result in underflow issues in the
 * dependent code.
 */
template <typename T> __device__ inline int find_first_set_bit(T v);

template <> __device__ inline int find_first_set_bit<Mask32>(Mask32 v) {
  return __ffs(v);
}

template <> __device__ inline int find_first_set_bit<Mask64>(Mask64 v) {
  return __ffsll(v);
}

/**
 * \return Number of bits set to 1.
 * \note Return value type matches that of the underlying device builtin.
 *       Changing to unsigned type may result in underflow issues in the
 * dependent code.
 */
template <typename T> __device__ inline int num_set_bits(T v);

template <> __device__ inline int num_set_bits<Mask32>(Mask32 v) {
  return __popc(v);
}

template <> __device__ inline int num_set_bits<Mask64>(Mask64 v) {
  return __popcll(v);
}

// Warp-level communication

/**
 * Ballot (warp vote function) with full mask (all bits are "1", i.e. all lanes
 * participate).
 *
 * \param[in] t the warp lane id, i.e. threadIdx.x % warpsize
 * \param[in] predicate the predicate to use for active lanes.
 * \note All of a warp's threads must be participate in this operation on AMD
 * architectures as well as NVIDIA Pascal and older NVIDIA architectures.
 */
template <typename MaskT> __device__ MaskT BALLOT1(int predicate);

#if (__CUDACC_VER_MAJOR__ >= 9)
#define INDEPENDENT_THREAD_SCHEDULING
#endif
#ifdef INDEPENDENT_THREAD_SCHEDULING
template <typename T> __device__ inline T SHFL10(T v) {
  return __shfl_sync(~0, v, 0);
}
template <typename T> __device__ inline T SHFL1(T v, uint32_t t) {
  return __shfl_sync(~0, v, t);
}
template <typename T, typename MaskT>
__device__ inline T SHFL1_XOR(T v, const MaskT m) {
  return __shfl_xor_sync(~0, v, m);
}
__device__ inline void SYNCWARP() { return __syncwarp(); }
template <> __device__ inline Mask32 BALLOT1(int predicate) {
  return __ballot_sync(~0, predicate);
}

// #  define SHFL10(v)        __shfl_sync(~0, (v), 0)
// #  define SHFL1(v, t)      __shfl_sync(~0, (v), (t))
// #  define SHFL1_XOR(v, m)  __shfl_xor_sync(~0, (v), (m))
// #  define SYNCWARP()       __syncwarp()
// #  define BALLOT1(v)       __ballot_sync(~0, (v))
#else
template <typename T> __device__ inline T SHFL10(T v) { return __shfl(v, 0); }
template <typename T> __device__ inline T SHFL1(T v, uint32_t t) {
  return __shfl(v, t);
}
template <typename T, typename MaskT>
__device__ inline T SHFL1_XOR(T v, const MaskT m) {
  return __shfl_xor(v, m);
}
__device__ inline void SYNCWARP() {
  /* sync/barrier all threads in a warp */
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
  __builtin_amdgcn_wave_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}
// BALLOT1 specializations
template <> __device__ inline Mask32 BALLOT1(int predicate) {
#ifdef __HIP_PLATFORM_AMD__
  // NOTE(HIP/AMD): __ballot returns an 64-bit integer on both warpsize 32 and
  // 64 architectures.
  auto mask64 = __ballot(predicate);
  return *reinterpret_cast<Mask32 *>(&mask64);
#else
  return __ballot(predicate);
#endif
}
template <> __device__ inline Mask64 BALLOT1(int predicate) {
  return __ballot(predicate);
}

// #  define SHFL10(v)        __shfl((v), 0)
// #  define SHFL1(v, t)      __shfl((v), (t))
// #  define SHFL1_XOR(v, m)  __shfl_xor((v), (m))
// #  define SYNCWARP()
// #  define BALLOT1(v)       __ballot((v))
#endif

#if (__CUDA_ARCH__ >= 700)
#define NANOSLEEP(d) __nanosleep((d))
#else
//: includes the __HIP_PLATFORM_AMD__ case
#define NANOSLEEP(d) clock()
#endif

template <typename T> inline __device__ T WarpReduceSum2(T acc) {
  return acc + SHFL1_XOR(acc, 1);
}
template <typename T> inline __device__ T WarpReduceSum4(T acc) {
  acc = WarpReduceSum2(acc);
  return acc + SHFL1_XOR(acc, 2);
}
template <typename T> inline __device__ T WarpReduceSum8(T acc) {
  acc = WarpReduceSum4(acc);
  return acc + SHFL1_XOR(acc, 4);
}
template <typename T> inline __device__ T WarpReduceSum16(T acc) {
  acc = WarpReduceSum8(acc);
  return acc + SHFL1_XOR(acc, 8);
}
template <typename T> inline __device__ T WarpReduceSum32(T acc) {
  acc = WarpReduceSum16(acc);
  return acc + SHFL1_XOR(acc, 16);
}
template <typename T> inline __device__ T WarpReduceSum64(T acc) {
  acc = WarpReduceSum32(acc);
  return acc + SHFL1_XOR(acc, 32);
}

template <typename T> inline __device__ T WarpReducePos2_ws32(T pos, int t) {
  T tmp = SHFL1(pos, t & 0x1e);
  pos += (t & 1) ? tmp : 0;
  return pos;
}
template <typename T> inline __device__ T WarpReducePos4_ws32(T pos, int t) {
  T tmp;
  pos = WarpReducePos2_ws32(pos, t);
  tmp = SHFL1(pos, (t & 0x1c) | 1);
  pos += (t & 2) ? tmp : 0;
  return pos;
}
template <typename T> inline __device__ T WarpReducePos8_ws32(T pos, int t) {
  T tmp;
  pos = WarpReducePos4_ws32(pos, t);
  tmp = SHFL1(pos, (t & 0x18) | 3);
  pos += (t & 4) ? tmp : 0;
  return pos;
}
template <typename T> inline __device__ T WarpReducePos16_ws32(T pos, int t) {
  T tmp;
  pos = WarpReducePos8_ws32(pos, t);
  tmp = SHFL1(pos, (t & 0x10) | 7);
  pos += (t & 8) ? tmp : 0;
  return pos;
}
template <typename T> inline __device__ T WarpReducePos32_ws32(T pos, int t) {
  T tmp;
  pos = WarpReducePos16_ws32(pos, t);
  tmp = SHFL1(pos, 0xf);
  pos += (t & 16) ? tmp : 0;
  return pos;
}

template <typename T> inline __device__ T WarpReducePos2_ws64(T pos, int t) {
  T tmp = SHFL1(pos, t & 0x3e);
  pos += (t & 1) ? tmp : 0;
  return pos;
}
template <typename T> inline __device__ T WarpReducePos4_ws64(T pos, int t) {
  T tmp;
  pos = WarpReducePos2_ws64(pos, t);
  tmp = SHFL1(pos, (t & 0x3c) | 1);
  pos += (t & 2) ? tmp : 0;
  return pos;
}
template <typename T> inline __device__ T WarpReducePos8_ws64(T pos, int t) {
  T tmp;
  pos = WarpReducePos4_ws64(pos, t);
  tmp = SHFL1(pos, (t & 0x38) | 3);
  pos += (t & 4) ? tmp : 0;
  return pos;
}
template <typename T> inline __device__ T WarpReducePos16_ws64(T pos, int t) {
  T tmp;
  pos = WarpReducePos8_ws64(pos, t);
  tmp = SHFL1(pos, (t & 0x30) | 7);
  pos += (t & 8) ? tmp : 0;
  return pos;
}
template <typename T> inline __device__ T WarpReducePos32_ws64(T pos, int t) {
  T tmp;
  pos = WarpReducePos16_ws64(pos, t);
  tmp = SHFL1(pos, (t & 0x20) | 15);
  pos += (t & 16) ? tmp : 0;
  return pos;
}
template <typename T> inline __device__ T WarpReducePos64_ws64(T pos, int t) {
  T tmp;
  pos = WarpReducePos32_ws64(pos, t);
  tmp = SHFL1(pos, 0x1f);
  pos += (t & 32) ? tmp : 0;
  return pos;
}

template <int warpsize> class WarpReduce {
public:
  template <typename T>
  static __device__ inline T prefix_sum(int t, T thread_value);
  template <typename T> static __device__ inline T sum(int t, T thread_value);
};

// explicit WarpReduce instantiations

template <> class WarpReduce<32> {
public:
  template <typename T>
  static __device__ inline T prefix_sum(int t, T thread_value) {
    return WarpReducePos32_ws32(thread_value, t);
  }
  template <typename T> static __device__ inline T sum(int t, T thread_value) {
    return WarpReduceSum32(thread_value);
  }
};

template <> class WarpReduce<64> {
public:
  template <typename T>
  static __device__ inline T prefix_sum(int t, T thread_value) {
    return WarpReducePos64_ws64(thread_value, t);
  }
  template <typename T> static __device__ inline T sum(int t, T thread_value) {
    return WarpReduceSum64(thread_value);
  }
};

///////////////////////////////////
// From src/SnappyBlockUtils.cu/hip
///////////////////////////////////

#if 0
/**
 * \note (07/12/23) unused
 */
inline __device__ double Int128ToDouble_rn(uint64_t lo, int64_t hi)
{
    double sign;
    if (hi < 0) {
        sign = -1.0;
        lo = (~lo) + 1;
        hi = (~hi) + (lo == 0);
    } else {
        sign = 1.0;
    }
    return sign * __fma_rn(__ll2double_rn(hi), 4294967296.0 * 4294967296.0, __ull2double_rn(lo));
}
#endif

/**
 * \note (07/12/23) used in:
 *   \verbatim
 *   src\SnappyKernels.cu
 *   191,44:     uint32_t data32           = (valid4) ? unaligned_load32(src +
 * pos + t) : 0; 205,69:           (offset < pos && offset + MAX_COPY_DISTANCE
 * >= pos + t && unaligned_load32(src + offset) == data32);
 *   \endverbatim
 */
/**
 * Loads a uint32_t that is not located at a 4-byte-multiple adress but at an
 * 1-byte-multiple address.
 */
inline __device__ uint32_t
unaligned_load32(const uint8_t *p) { // points to uint8_t but is pointer, *p = {
                                     // *p7.*p6.*p5.*p4.*p3.*p2.*p1.*p0 }
  uint32_t ofs =
      3 & reinterpret_cast<uintptr_t>(
              p); // offset: models pointer as int, masks in 2 least-sig. bits,
                  // converts to 4 byte uint: 00h...[00h.00h.00h{0.0.0.0.p1.p0}]
                  // this is the offset from the previous 4-byte-aligned value
                  // p % 4 == 0 => ofs == 0
  const uint32_t *p32 = reinterpret_cast<const uint32_t *>(
      p -
      ofs); // substracts offset from original pointer, casts result to const
            // uint32_t ptr this is the address of previous 4-byte-aligned value
  uint32_t v = p32[0]; // v is dereferenced p32, i.e. the pevious 4-byte-aligned
                       // value, &v = p - {p1.p0}
  return (ofs) ? __funnelshift_r(v, p32[1], ofs * 8)
               : v; // if the offset is 0, returns v. Otherwise, needs to
                    // combine the result from previous 4-byte word and
                    // following 4-byte word Note: ofs*8 is ofs in bits.
  // __device__​ unsigned int 	__funnelshift_r ( unsigned int  lo, unsigned
  // int  hi, unsigned int  shift )
  //  Concatenate hi : lo -> shift right by (shift & 31 bits) -> return the
  //  least significant 32 bits (4 byte).
}

#if 0
/**
 * \note (07/12/23) unused
 */
inline __device__ uint64_t unaligned_load64(const uint8_t *p) {
  uint32_t ofs = 3 & reinterpret_cast<uintptr_t>(p);
  const uint32_t *p32 = reinterpret_cast<const uint32_t *>(p - ofs);
  uint32_t v0 = p32[0];
  uint32_t v1 = p32[1];
  if (ofs) {
    v0 = __funnelshift_r(v0, v1, ofs * 8);
    v1 = __funnelshift_r(v1, p32[2], ofs * 8);
  }
  return (((uint64_t)v1) << 32) | v0;
}

/**
 * \note (07/12/23) unused
 */
template<unsigned int nthreads, bool sync_before_store>
inline __device__ void memcpy_block(void *dstv, const void *srcv, uint32_t len, uint32_t t)
{
    uint8_t *dst = reinterpret_cast<uint8_t *>(dstv);
    const uint8_t *src = reinterpret_cast<const uint8_t *>(srcv);
    uint32_t dst_align_bytes, src_align_bytes, src_align_bits;
    // Align output to 32-bit
    dst_align_bytes = 3 & -reinterpret_cast<intptr_t>(dst);
    if (dst_align_bytes != 0) {
        uint32_t align_len = min(dst_align_bytes, len);
        uint8_t b;
        if (t < align_len) {
            b = src[t];
        }
        if (sync_before_store) {
            __syncthreads();
        }
        if (t < align_len) {
            dst[t] = b;
        }
        src += align_len;
        dst += align_len;
        len -= align_len;
    }
    src_align_bytes = (uint32_t)(3 & reinterpret_cast<uintptr_t>(src));
    src_align_bits = src_align_bytes * 8;
    while (len >= 4) {
        const uint32_t *src32 = reinterpret_cast<const uint32_t *>(src - src_align_bytes);
        uint32_t copy_cnt = min(len >> 2, nthreads);
        uint32_t v;
        if (t < copy_cnt) {
            v = src32[t];
            if (src_align_bits != 0) {
                v = __funnelshift_r(v, src32[t + 1], src_align_bits);
            }
        }
        if (sync_before_store) {
            __syncthreads();
        }
        if (t < copy_cnt) {
            reinterpret_cast<uint32_t *>(dst)[t] = v;
        }
        src += copy_cnt * 4;
        dst += copy_cnt * 4;
        len -= copy_cnt * 4;
    }
    if (len != 0) {
        uint8_t b;
        if (t < len) {
            b = src[t];
        }
        if (sync_before_store) {
            __syncthreads();
        }
        if (t < len) {
            dst[t] = b;
        }
    }
}

/**
 * @brief Compares two strings
 */
/**
 * \note (07/12/23) unused
 */
template<class T, const T lesser, const T greater, const T equal>
inline __device__ T nvstr_compare(const char *as, uint32_t alen, const char *bs, uint32_t blen)
{
    uint32_t len = min(alen, blen);
    uint32_t i = 0;
    if (len >= 4) {
        uint32_t align_a = 3 & reinterpret_cast<uintptr_t>(as);
        uint32_t align_b = 3 & reinterpret_cast<uintptr_t>(bs);
        const uint32_t *as32 = reinterpret_cast<const uint32_t *>(as - align_a);
        const uint32_t *bs32 = reinterpret_cast<const uint32_t *>(bs - align_b);
        uint32_t ofsa = align_a * 8;
        uint32_t ofsb = align_b * 8;
        do {
            uint32_t a = *as32++;
            uint32_t b = *bs32++;
            if (ofsa)
                a = __funnelshift_r(a, *as32, ofsa);
            if (ofsb)
                b = __funnelshift_r(b, *bs32, ofsb);
            if (a != b) {
                return (lesser == greater || __byte_perm(a, 0, 0x0123) < __byte_perm(b, 0, 0x0123)) ? lesser : greater;
            }
            i += 4;
        } while (i + 4 <= len);
    }
    while (i < len) {
        uint8_t a = as[i];
        uint8_t b = bs[i];
        if (a != b) {
            return (a < b) ? lesser : greater;
        }
        ++i;
    }
    return (alen == blen) ? equal : (alen < blen) ? lesser : greater;
}

/**
 * \note (07/12/23) unused
 */
inline __device__ bool nvstr_is_lesser(const char *as, uint32_t alen, const char *bs, uint32_t blen)
{
    return nvstr_compare<bool, true, false, false>(as, alen, bs, blen);
}

/**
 * \note (07/12/23) unused
 */
inline __device__ bool nvstr_is_greater(const char *as, uint32_t alen, const char *bs, uint32_t blen)
{
    return nvstr_compare<bool, false, true, false>(as, alen, bs, blen);
}

/**
 * \note (07/12/23) unused
 */
inline __device__ bool nvstr_is_equal(const char *as, uint32_t alen, const char *bs, uint32_t blen)
{
    return nvstr_compare<bool, false, false, true>(as, alen, bs, blen);
}
#endif

} // namespace hipcomp
