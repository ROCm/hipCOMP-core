/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include "cstdint"
#include "device_functions.cuh"
#include "hip/hip_runtime.h"

namespace hipcomp {
namespace deflate {

constexpr int max_bits = 15;     // maximum bits in a code
constexpr int max_l_codes = 286; // maximum number of literal/length codes
constexpr int max_d_codes = 30;  // maximum number of distance codes
constexpr int fix_l_codes = 288; // number of fixed literal/length codes

constexpr int log2_len_lut = 10;
constexpr int log2_dist_lut = 8;

/**
 * @brief Intermediate arrays for building huffman tables
 */
struct scratch_arr {
  int16_t lengths[max_l_codes + max_d_codes]; ///< descriptor code lengths
  int16_t
      offs[max_bits + 1]; ///< offset in symbol table for each length (scratch)
};

/**
 * @brief Huffman LUTs for length and distance codes
 */
struct lut_arr {
  int32_t lenlut[1 << log2_len_lut];   ///< LUT for length decoding
  int32_t distlut[1 << log2_dist_lut]; ///< LUT for fast distance decoding
};

/**
 * @brief Inter-warp communication queue
 */
template <int BATCH_SIZE, int BATCH_COUNT> struct xwarp_s {
  int32_t batch_len[BATCH_COUNT]; //< Length of each batch - <0:end, 0:not
                                  // ready, >0:symbol count
  union {
    uint32_t symqueue[BATCH_COUNT * BATCH_SIZE];
    uint8_t symqueue8[BATCH_COUNT * BATCH_SIZE * 4];
  } u;
};

/// @brief Prefetcher state
template <int prefetch_size> struct prefetch_queue_s {
  static constexpr int PREFETCH_SIZE = prefetch_size;

  uint8_t const *cur_p; ///< Prefetch location
  int run;              ///< prefetcher will exit when run=0
  uint8_t pref_data[prefetch_size];
};

template <typename T, typename PREFETCH_QUEUE_S>
__device__ inline volatile uint32_t *
prefetch_addr32(volatile PREFETCH_QUEUE_S &q, T *ptr) {
  return reinterpret_cast<volatile uint32_t *>(
      &q.pref_data[(PREFETCH_QUEUE_S::PREFETCH_SIZE - 4) & (size_t)(ptr)]);
}

/// permutation of code length codes
static const __device__ __constant__ uint8_t g_code_order[19 + 1] = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15, 0xff};

/// permutation of code length codes
static const __device__ __constant__ uint16_t
    g_lens[29] = { // Size base for length codes 257..285
        3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
        31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};

static const __device__ __constant__ uint16_t
    g_lext[29] = { // Extra bits for length codes 257..285
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
        2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};

static const __device__ __constant__ uint16_t
    g_dists[30] = { // Offset base for distance codes 0..29
        1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
        33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
        1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const __device__ __constant__ uint16_t
    g_dext[30] = { // Extra bits for distance codes 0..29
        0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
        6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

/**
 * @brief Inflate decompressor state
 */
template <int batch_size, int batch_count, int prefetch_size>
class inflate_state_s {
private:
  /**
   * @brief Given the list of code lengths length[0..n-1] representing a
   * canonical Huffman code for n symbols, construct the tables required to
   * decode those codes.  Those tables are the number of codes of each length,
   * and the symbols sorted by length, retaining their original order within
   * each length.  The return value is zero for a complete code set, negative
   * for an over- subscribed code set, and positive for an incomplete code set.
   * The tables can be used if the return value is zero or positive, but they
   * cannot be used if the return value is negative.  If the return value is
   * zero, it is not possible for decode() using that table to return an
   * error--any stream of enough bits will resolve to a symbol.  If the return
   * value is positive, then it is possible for decode() using that table to
   * return an error for received codes past the end of the incomplete lengths.
   *
   * Not used by decode(), but used for error checking, count[0] is the number
   * of the n symbols not in the code.  So n - count[0] is the number of
   * codes.  This is useful for checking for incomplete codes that have more
   * than one symbol, which is an error in a dynamic block.
   *
   * Assumption: for all i in 0..n-1, 0 <= length[i] <= max_bits
   * This is assured by the construction of the length arrays in dynamic() and
   * fixed() and is not verified by construct().
   *
   * Format notes:
   *
   * - Permitted and expected examples of incomplete codes are one of the fixed
   *   codes and any code with a single symbol which in deflate is coded as one
   *   bit instead of zero bits.  See the format notes for fixed() and
   * dynamic().
   *
   * - Within a given code length, the symbols are kept in ascending order for
   *   the code bits definition.
   */
  __device__ inline int construct(int16_t *counts, int16_t *symbols,
                                  int16_t const *length, int n) {
    int symbol; // current symbol when stepping through length[]
    int len;    // current length when stepping through counts[]
    int left;   // number of possible codes left of current length
    int16_t *offs = this->u.scratch.offs;

    // count number of codes of each length
    for (len = 0; len <= max_bits; len++)
      counts[len] = 0;
    for (symbol = 0; symbol < n; symbol++)
      (counts[length[symbol]])++; // assumes lengths are within bounds
    if (counts[0] == n)           // no codes!
      return 0;                   // complete, but decode() will fail

    // check for an over-subscribed or incomplete set of lengths
    left = 1; // one possible code of zero length
    for (len = 1; len <= max_bits; len++) {
      left <<= 1;          // one more bit, double codes left
      left -= counts[len]; // deduct count from possible codes
      if (left < 0)
        return left; // over-subscribed--return negative
    } // left > 0 means incomplete

    // generate offsets into symbol table for each length for sorting
    offs[1] = 0;
    for (len = 1; len < max_bits; len++)
      offs[len + 1] = offs[len] + counts[len];

    // put symbols in table sorted by length, by symbol order within each length
    for (symbol = 0; symbol < n; symbol++)
      if (length[symbol] != 0)
        symbols[offs[length[symbol]]++] = symbol;

    // return zero for complete set, positive for incomplete set
    return left;
  }

public:
  static constexpr int BATCH_SIZE = batch_size;
  static constexpr int BATCH_COUNT = batch_count;
  static constexpr int PREFETCH_SIZE = prefetch_size;
  using XWARP_S = xwarp_s<batch_size, batch_count>;
  using PREFETCH_QUEUE_S = prefetch_queue_s<prefetch_size>;

  // output state
  uint8_t *out;     ///< output buffer
  uint8_t *outbase; ///< start of output buffer
  uint8_t *outend;  ///< end of output buffer
  // Input state
  uint8_t const *cur; ///< input buffer
  uint8_t const *end; ///< end of input buffer

  uint2 bitbuf;    ///< bit buffer (64-bit)
  uint32_t bitpos; ///< position in bit buffer

  int32_t err;             ///< Error status
  int btype;               ///< current block type
  int blast;               ///< last block
  uint32_t stored_blk_len; ///< length of stored (uncompressed) block

  uint16_t first_slow_len; ///< first code not in fast LUT
  uint16_t index_slow_len;
  uint16_t first_slow_dist;
  uint16_t index_slow_dist;

  volatile XWARP_S x;
  volatile PREFETCH_QUEUE_S pref;

  int16_t lencnt[max_bits + 1];
  int16_t lensym[fix_l_codes]; // Assumes fix_l_codes >= max_l_codes
  int16_t distcnt[max_bits + 1];
  int16_t distsym[max_d_codes];

  union {
    scratch_arr scratch;
    lut_arr lut;
  } u;

  // functions

  __device__ inline void init_prefetcher(int t) volatile {
    if (t == 0) {
      this->pref.cur_p = this->cur;
      this->pref.run = 1;
    }
  }

  __device__ inline uint32_t showbits(uint32_t n) {
    uint32_t next32 =
        __funnelshift_rc(this->bitbuf.x, this->bitbuf.y, this->bitpos);
    return (next32 & ((1 << n) - 1));
  }

  __device__ inline uint32_t nextbits32() {
    return __funnelshift_rc(this->bitbuf.x, this->bitbuf.y, this->bitpos);
  }

  __device__ inline void skipbits(int32_t n) {
    uint32_t bitpos = this->bitpos + n;
    if (bitpos >= 32) {
      auto cur = this->cur + 8;
      this->bitbuf.x = this->bitbuf.y;
      this->bitbuf.y =
          (cur < this->end) ? *reinterpret_cast<uint32_t const *>(cur) : 0;
      this->cur = cur - 4;
      bitpos &= 0x1f;
    }
    this->bitpos = bitpos;
  }

  // TODO: If we require 4-byte alignment of input bitstream & length (padded),
  // reading bits would become quite a bit faster
  __device__ inline uint32_t getbits(uint32_t n) {
    uint32_t v = this->showbits(n);
    this->skipbits(n);
    return v;
  }

  /**
   * @brief Initializes a stored block.
   *
   * Format notes:
   *
   * - After the two-bit stored block type (00), the stored block length and
   *   stored bytes are byte-aligned for fast copying.  Therefore any leftover
   *   bits in the byte that has the last bit of the type, as many as seven, are
   *   discarded.  The value of the discarded bits are not defined and should
   * not be checked against any expectation.
   *
   * - The second inverted copy of the stored block length does not have to be
   *   checked, but it's probably a good idea to do so anyway.
   *
   * - A stored block can have zero length.  This is sometimes used to
   * byte-align subsets of the compressed data for random access or partial
   * recovery.
   */
  __device__ inline int init_stored() {
    uint32_t len, nlen; // length of stored block

    // Byte align
    if (this->bitpos & 7) {
      this->skipbits(8 - (this->bitpos & 7));
    }
    if (this->cur + (this->bitpos >> 3) >= this->end) {
      return 2; // Not enough input
    }
    // get length and check against its one'this complement
    len = this->getbits(16);
    nlen = this->getbits(16);
    if (len != (nlen ^ 0xffff)) {
      return -2; // didn't match complement!
    }
    if (this->cur + (this->bitpos >> 3) + len > this->end) {
      return 2; // Not enough input
    }
    this->stored_blk_len = len;

    // done with a valid stored block
    return 0;
  }

  /// Copy bytes from stored block to destination
  __device__ inline void copy_stored(int t) {
    auto len = this->stored_blk_len;
    auto cur = this->cur + this->bitpos / 8;
    auto out = this->out;
    auto outend = this->outend;
    auto const slow_bytes =
        static_cast<uint8_t>(min(len,
                                 (int)((16 - reinterpret_cast<size_t>(out)) %
                                       16))); // TODO(HIP/AMD): static_cast safe
                                              // here (WAR for compiler error)?

    // Slow copy until output is 16B aligned
    if (slow_bytes) {
      for (int i = t; i < slow_bytes; i += blockDim.x) {
        if (out + i < outend) {
          out[i] =
              cur[i]; // Input range has already been validated in init_stored()
        }
      }
      cur += slow_bytes;
      out += slow_bytes;
      len -= slow_bytes;
    }
    auto fast_bytes = len;
    if (out < outend) {
      fast_bytes = (int)min((size_t)fast_bytes, (outend - out));
    }
    fast_bytes &= ~0xf;
    auto bitpos = ((int)((size_t)cur % 4)) * 8;
    auto cur4 = cur - (bitpos / 8);
    if (out < outend) {
      // Fast copy 16 bytes at a time
      for (int i = t * 16; i < fast_bytes; i += blockDim.x * 16) {
        uint4 u;
        u.x = *reinterpret_cast<uint32_t const *>(cur4 + i + 0 * 4);
        u.y = *reinterpret_cast<uint32_t const *>(cur4 + i + 1 * 4);
        u.z = *reinterpret_cast<uint32_t const *>(cur4 + i + 2 * 4);
        u.w = *reinterpret_cast<uint32_t const *>(cur4 + i + 3 * 4);
        if (bitpos != 0) {
          uint32_t v =
              (bitpos != 0)
                  ? *reinterpret_cast<uint32_t const *>(cur4 + i + 4 * 4)
                  : 0;
          u.x = __funnelshift_rc(u.x, u.y, bitpos);
          u.y = __funnelshift_rc(u.y, u.z, bitpos);
          u.z = __funnelshift_rc(u.z, u.w, bitpos);
          u.w = __funnelshift_rc(u.w, v, bitpos);
        }
        *reinterpret_cast<uint4 *>(out + i) = u;
      }
    }
    cur += fast_bytes;
    out += fast_bytes;
    len -= fast_bytes;
    // Slow copy for remaining bytes
    for (int i = t; i < len; i += blockDim.x) {
      if (out + i < outend) {
        out[i] =
            cur[i]; // Input range has already been validated in init_stored()
      }
    }
    out += len;
    __syncthreads();
    if (t == 0) {
      // Reset bitstream to end of block
      auto p = cur + len;
      auto prefix_bytes = (uint32_t)(((size_t)p) & 3);
      p -= prefix_bytes;
      this->cur = p;
      this->bitbuf.x =
          (p < this->end) ? *reinterpret_cast<uint32_t const *>(p) : 0;
      p += 4;
      this->bitbuf.y =
          (p < this->end) ? *reinterpret_cast<uint32_t const *>(p) : 0;
      this->bitpos = prefix_bytes * 8;
      this->out = out;
    }
  }

  /**
   * @brief Build lookup tables for faster decode
   * LUT format is symbols*16+length
   */
  __device__ void init_length_lut(int t) {
    int32_t *lut = this->u.lut.lenlut;

    for (uint32_t bits = t; bits < (1 << log2_len_lut); bits += blockDim.x) {
      int16_t const *cnt = this->lencnt;
      int16_t const *symbols = this->lensym;
      int sym =
          -320; // TODO(HIP/AMD): -10 << 5; the original left shift is not
                // accepted by hipclang as it is undefined (see
                // https://en.cppreference.com/w/cpp/language/operator_arithmetic)
      unsigned int first = 0;
      unsigned int rbits = __brev(bits) >> (32 - log2_len_lut);
      for (unsigned int len = 1; len <= log2_len_lut; len++) {
        unsigned int code = (rbits >> (log2_len_lut - len)) - first;
        unsigned int count = cnt[len];
        if (code < count) {
          sym = symbols[code];
          if (sym > 256) {
            int lext = g_lext[sym - 257];
            sym = (256 + g_lens[sym - 257]) | (((1 << lext) - 1) << (16 - 5)) |
                  (len << (24 - 5));
            len += lext;
          }
          sym = (sym << 5) | len;
          break;
        }
        symbols += count; // else update for next length
        first += count;
        first <<= 1;
      }
      lut[bits] = sym;
    }
    if (!t) {
      unsigned int first = 0;
      unsigned int index = 0;
      int16_t const *cnt = this->lencnt;
      for (unsigned int len = 1; len <= log2_len_lut; len++) {
        unsigned int count = cnt[len];
        index += count;
        first += count;
        first <<= 1;
      }
      this->first_slow_len = first;
      this->index_slow_len = index;
    }
  }

  /**
   * @brief Build lookup tables for faster decode of distance symbol
   * LUT format is symbols*16+length
   */
  __device__ void init_distance_lut(int t) {
    int32_t *lut = this->u.lut.distlut;

    for (uint32_t bits = t; bits < (1 << log2_dist_lut); bits += blockDim.x) {
      int16_t const *cnt = this->distcnt;
      int16_t const *symbols = this->distsym;
      int sym = 0;
      unsigned int first = 0;
      unsigned int rbits = __brev(bits) >> (32 - log2_dist_lut);
      for (unsigned int len = 1; len <= log2_dist_lut; len++) {
        unsigned int code = (rbits >> (log2_dist_lut - len)) - first;
        unsigned int count = cnt[len];
        if (code < count) {
          int dist = symbols[code];
          int dext = g_dext[dist];
          sym = g_dists[dist] | (dext << 15);
          sym = (sym << 5) | len;
          break;
        }
        symbols += count; // else update for next length
        first += count;
        first <<= 1;
      }
      lut[bits] = sym;
    }
    if (!t) {
      unsigned int first = 0;
      unsigned int index = 0;
      int16_t const *cnt = this->distcnt;
      for (unsigned int len = 1; len <= log2_dist_lut; len++) {
        unsigned int count = cnt[len];
        index += count;
        first += count;
        first <<= 1;
      }
      this->first_slow_dist = first;
      this->index_slow_dist = index;
    }
  }

  /**
   * @brief Decode a code from the stream s using huffman table
   * {symbols,counts}. Return the symbol or a negative value if there is an
   * error. If all of the lengths are zero, i.e. an empty code, or if the code
   * is incomplete and an invalid code is received, then -10 is returned after
   * reading max_bits bits.
   *
   * Format notes:
   *
   * - The codes as stored in the compressed data are bit-reversed relative to
   *   a simple integer ordering of codes of the same lengths.  Hence below the
   *   bits are pulled from the compressed data one at a time and used to
   *   build the code value reversed from what is in the stream in order to
   *   permit simple integer comparisons for decoding.  A table-based decoding
   *   scheme (as used in zlib) does not need to do this reversal.
   *
   * - The first code for the shortest length is all zeros.  Subsequent codes of
   *   the same length are simply integer increments of the previous code.  When
   *   moving up a length, a zero bit is appended to the code.  For a complete
   *   code, the last code of the longest length will be all ones.
   *
   * - Incomplete codes are handled by this decoder, since they are permitted
   *   in the deflate format.  See the format notes for fixed() and dynamic().
   */
  __device__ inline int decode(int16_t const *counts, int16_t const *symbols) {
    unsigned int len;   // current number of bits in code
    unsigned int code;  // len bits being decoded
    unsigned int first; // first code of length len
    unsigned int count; // number of codes of length len
    uint32_t next32r = __brev(this->nextbits32());

    first = 0;
    for (len = 1; len <= max_bits; len++) {
      code = (next32r >> (32 - len)) - first;
      count = counts[len];
      if (code < count) // if length len, return symbol
      {
        this->skipbits(len);
        return symbols[code];
      }
      symbols += count; // else update for next length
      first += count;
      first <<= 1;
    }
    return -10; // ran out of codes
  }

  /// Dynamic block (custom huffman tables)
  __device__ inline int init_dynamic() {
    int nlen, ndist, ncode; /* number of lengths in descriptor */
    int index;              /* index of lengths[] */
    int err;                /* construct() return value */
    int16_t *lengths = this->u.scratch.lengths;

    // get number of lengths in each table, check lengths
    nlen = this->getbits(5) + 257;
    ndist = this->getbits(5) + 1;
    ncode = this->getbits(4) + 4;
    if (nlen > max_l_codes || ndist > max_d_codes) {
      return -3; // bad counts
    }
    // read code length code lengths (really), missing lengths are zero
    for (index = 0; index < ncode; index++)
      lengths[g_code_order[index]] = this->getbits(3);
    for (; index < 19; index++)
      lengths[g_code_order[index]] = 0;

    // build huffman table for code lengths codes (use lencode temporarily)
    err = this->construct(this->lencnt, this->lensym, lengths, 19);
    if (err != 0) // require complete code set here
      return -4;

    // read length/literal and distance code length tables
    index = 0;
    while (index < nlen + ndist) {
      int symbol = this->decode(this->lencnt, this->lensym);
      if (symbol < 0)
        return symbol; // invalid symbol
      if (symbol < 16) // length in 0..15
        lengths[index++] = symbol;
      else {                // repeat instruction
        int len = 0;        // last length to repeat, assume repeating zeros
        if (symbol == 16) { // repeat last length 3..6 times
          if (index == 0)
            return -5;              // no last length!
          len = lengths[index - 1]; // last length
          symbol = 3 + this->getbits(2);
        } else if (symbol == 17) // repeat zero 3..10 times
          symbol = 3 + this->getbits(3);
        else // == 18, repeat zero 11..138 times
          symbol = 11 + this->getbits(7);
        if (index + symbol > nlen + ndist)
          return -6;     // too many lengths!
        while (symbol--) // repeat last or zero symbol times
          lengths[index++] = len;
      }
    }

    // check for end-of-block code -- there better be one!
    if (lengths[256] == 0)
      return -9;

    // build huffman table for literal/length codes
    err = this->construct(this->lencnt, this->lensym, lengths, nlen);
    if (err && (err < 0 || nlen != this->lencnt[0] + this->lencnt[1]))
      return -7; // incomplete code ok only for single length 1 code

    // build huffman table for distance codes
    err = this->construct(this->distcnt, this->distsym, &lengths[nlen], ndist);
    if (err && (err < 0 || ndist != this->distcnt[0] + this->distcnt[1]))
      return -8; // incomplete code ok only for single length 1 code

    return 0;
  }

  /**
   * @brief Initializes a fixed codes block.
   *
   * Format notes:
   *
   * - This block type can be useful for compressing small amounts of data for
   *   which the size of the code descriptions in a dynamic block exceeds the
   *   benefit of custom codes for that block.  For fixed codes, no bits are
   *   spent on code descriptions.  Instead the code lengths for literal/length
   *   codes and distance codes are fixed.  The specific lengths for each symbol
   *   can be seen in the "for" loops below.
   *
   * - The literal/length code is complete, but has two symbols that are invalid
   *   and should result in an error if received.  This cannot be implemented
   *   simply as an incomplete code since those two symbols are in the "middle"
   *   of the code.  They are eight bits long and the longest literal/length\
   *   code is nine bits.  Therefore the code must be constructed with those
   *   symbols, and the invalid symbols must be detected after decoding.
   *
   * - The fixed distance codes also have two invalid symbols that should result
   *   in an error if received.  Since all of the distance codes are the same
   *   length, this can be implemented as an incomplete code.  Then the invalid
   *   codes are detected while decoding.
   */
  __device__ inline int init_fixed() {
    int16_t *lengths = this->u.scratch.lengths;
    int symbol;

    // literal/length table
    for (symbol = 0; symbol < 144; symbol++)
      lengths[symbol] = 8;
    for (; symbol < 256; symbol++)
      lengths[symbol] = 9;
    for (; symbol < 280; symbol++)
      lengths[symbol] = 7;
    for (; symbol < fix_l_codes; symbol++)
      lengths[symbol] = 8;
    this->construct(this->lencnt, this->lensym, lengths, fix_l_codes);

    // distance table
    for (symbol = 0; symbol < max_d_codes; symbol++)
      lengths[symbol] = 5;

    // build huffman table for distance codes
    this->construct(this->distcnt, this->distsym, lengths, max_d_codes);

    return 0;
  }
};

} // namespace deflate
} // namespace hipcomp
