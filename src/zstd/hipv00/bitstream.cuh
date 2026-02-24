/* ******************************************************************
 * bitstream
 * Part of FSE library
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * You can contact the author at :
 * - Source repository : https://github.com/Cyan4973/FiniteStateEntropy
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 ****************************************************************** */

// MIT License
//
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights
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

#include "common.cuh"

namespace hipcomp {
namespace zstd {

typedef size_t BitContainerType;

/* bitStream can mix input from multiple sources.
 * A critical property of these streams is that they encode and decode in
 * **reverse** direction. So the first bit sequence you add will be the last to
 * be read, like a LIFO stack.
 */
/*-******************************************
 *  bitStream encoding stream (write forward)
 ********************************************/
typedef struct {
  BitContainerType bitContainer;
  unsigned bitPos;
  char *startPtr;
  char *ptr;
  char *endPtr;
} BIT_CStream_t;

/* Start with initCStream, providing the size of buffer to write into.
 *  bitStream will never write outside of this buffer.
 *  `dstCapacity` must be >= sizeof(bitD->bitContainer), otherwise @return will
 * be an error code.
 *
 *  bits are first added to a local register.
 *  Local register is BitContainerType, 64-bits on 64-bits systems, or 32-bits
 * on 32-bits systems. Writing data into memory is an explicit operation,
 * performed by the flushBits function. Hence keep track how many bits are
 * potentially stored into local register to avoid register overflow. After a
 * flushBits, a maximum of 7 bits might still be stored into local register.
 *
 *  Avoid storing elements of more than 24 bits if you want compatibility with
 * 32-bits bitstream readers.
 *
 *  Last operation is to close the bitStream.
 *  The function returns the final size of CStream in bytes.
 *  If data couldn't fit into `dstBuffer`, it will return a 0 ( == not storable)
 */

/*-******************************************
 *  bitStream decoding stream (read backward)
 ********************************************/
typedef struct {
  BitContainerType bitContainer;
  unsigned bitsConsumed;
  const char *ptr;
  const char *start;
  const char *limitPtr;
} BIT_DStream_t;

/* Start by invoking BIT_initDStream().
 *  A chunk of the bitStream is then stored into a local register.
 *  Local register size is 64-bits on 64-bits systems, 32-bits on 32-bits
 * systems (BitContainerType). You can then retrieve bitFields stored into the
 * local register, **in reverse order**. Local register is explicitly reloaded
 * from memory by the BIT_reloadDStream() method. A reload guarantee a minimum
 * of ((8*sizeof(bitD->bitContainer))-7) bits when its result is
 * BIT_DStream_unfinished. Otherwise, it can be less than that, so proceed
 * accordingly. Checking if DStream has reached its end can be performed with
 * BIT_endOfDStream().
 */

typedef enum {
  BIT_DStream_unfinished = 0,  /* fully refilled */
  BIT_DStream_endOfBuffer = 1, /* still some bits left in bitstream */
  BIT_DStream_completed = 2,   /* bitstream entirely consumed, bit-exact */
  BIT_DStream_overflow =
      3               /* user requested more bits than present in bitstream */
} BIT_DStream_status; /* result of BIT_reloadDStream() */

/*=====    Local Constants   =====*/
static DEVICE_CONSTANT unsigned BIT_mask[] = {
    0,          1,         3,         7,         0xF,       0x1F,
    0x3F,       0x7F,      0xFF,      0x1FF,     0x3FF,     0x7FF,
    0xFFF,      0x1FFF,    0x3FFF,    0x7FFF,    0xFFFF,    0x1FFFF,
    0x3FFFF,    0x7FFFF,   0xFFFFF,   0x1FFFFF,  0x3FFFFF,  0x7FFFFF,
    0xFFFFFF,   0x1FFFFFF, 0x3FFFFFF, 0x7FFFFFF, 0xFFFFFFF, 0x1FFFFFFF,
    0x3FFFFFFF, 0x7FFFFFFF};
#define BIT_MASK_SIZE (sizeof(BIT_mask) / sizeof(BIT_mask[0]))

DEVICE_INLINE BitContainerType BIT_getLowerBits(BitContainerType bitContainer,
                                                U32 const nbBits) {
  return bitContainer & BIT_mask[nbBits];
}

DEVICE_INLINE void BIT_addBits(BIT_CStream_t *bitC, BitContainerType value,
                               unsigned nbBits) {
  (void)sizeof(char[((sizeof(BIT_mask) / sizeof(BIT_mask[0])) == 32) ? 1 : -1]);
  bitC->bitContainer |= BIT_getLowerBits(value, nbBits) << bitC->bitPos;
  bitC->bitPos += nbBits;
}

DEVICE_INLINE void BIT_flushBits(BIT_CStream_t *bitC) {
  size_t const nbBytes = bitC->bitPos >> 3;
  MEM_writeLEST(bitC->ptr, bitC->bitContainer);
  bitC->ptr += nbBytes;
  if (bitC->ptr > bitC->endPtr)
    bitC->ptr = bitC->endPtr;
  bitC->bitPos &= 7;
  bitC->bitContainer >>= nbBytes * 8;
}

/*! BIT_initDStream() :
 *  Initialize a BIT_DStream_t.
 * `bitD` : a pointer to an already allocated BIT_DStream_t structure.
 * `srcSize` must be the *exact* size of the bitStream, in bytes.
 * @return : size of stream (== srcSize), or an errorCode if a problem is
 * detected
 */
DEVICE_INLINE size_t BIT_initDStream(BIT_DStream_t *bitD, const void *srcBuffer,
                                     size_t srcSize) {
  if (srcSize < 1) {
    __builtin_memset((bitD), (0), (sizeof(*bitD)));
    return ((size_t)-ZSTD_error_srcSize_wrong);
  }

  bitD->start = (const char *)srcBuffer;
  bitD->limitPtr = bitD->start + sizeof(bitD->bitContainer);

  if (srcSize >= sizeof(bitD->bitContainer)) {
    bitD->ptr = (const char *)srcBuffer + srcSize - sizeof(bitD->bitContainer);
    bitD->bitContainer = MEM_readLEST(bitD->ptr);
    {
      BYTE const lastByte = ((const BYTE *)srcBuffer)[srcSize - 1];
      bitD->bitsConsumed = lastByte ? 8 - ZSTD_highbit32(lastByte) : 0;
      if (lastByte == 0)
        return ((size_t)-ZSTD_error_GENERIC);
    }
  } else {
    bitD->ptr = bitD->start;
    bitD->bitContainer = *(const BYTE *)(bitD->start);
    switch (srcSize) {
    case 7:
      bitD->bitContainer += (BitContainerType)(((const BYTE *)(srcBuffer))[6])
                            << (sizeof(bitD->bitContainer) * 8 - 16);
      ;
      __attribute__((__fallthrough__));

    case 6:
      bitD->bitContainer += (BitContainerType)(((const BYTE *)(srcBuffer))[5])
                            << (sizeof(bitD->bitContainer) * 8 - 24);
      ;
      __attribute__((__fallthrough__));

    case 5:
      bitD->bitContainer += (BitContainerType)(((const BYTE *)(srcBuffer))[4])
                            << (sizeof(bitD->bitContainer) * 8 - 32);
      ;
      __attribute__((__fallthrough__));

    case 4:
      bitD->bitContainer += (BitContainerType)(((const BYTE *)(srcBuffer))[3])
                            << 24;
      ;
      __attribute__((__fallthrough__));

    case 3:
      bitD->bitContainer += (BitContainerType)(((const BYTE *)(srcBuffer))[2])
                            << 16;
      ;
      __attribute__((__fallthrough__));

    case 2:
      bitD->bitContainer += (BitContainerType)(((const BYTE *)(srcBuffer))[1])
                            << 8;
      ;
      __attribute__((__fallthrough__));

    default:
      break;
    }
    {
      BYTE const lastByte = ((const BYTE *)srcBuffer)[srcSize - 1];
      bitD->bitsConsumed = lastByte ? 8 - ZSTD_highbit32(lastByte) : 0;
      if (lastByte == 0)
        return ((size_t)-ZSTD_error_corruption_detected);
    }
    bitD->bitsConsumed += (U32)(sizeof(bitD->bitContainer) - srcSize) * 8;
  }

  return srcSize;
}

DEVICE_INLINE BitContainerType BIT_getMiddleBits(BitContainerType bitContainer,
                                                 U32 const start,
                                                 U32 const nbBits) {
  U32 const regMask = sizeof(bitContainer) * 8 - 1;
  return (bitContainer >> (start & regMask)) & ((((U64)1) << nbBits) - 1);
}

DEVICE_INLINE BitContainerType BIT_lookBits(const BIT_DStream_t *bitD,
                                            U32 nbBits) {

  return BIT_getMiddleBits(
      bitD->bitContainer,
      (sizeof(bitD->bitContainer) * 8) - bitD->bitsConsumed - nbBits, nbBits);
}

DEVICE_INLINE BitContainerType BIT_lookBitsFast(const BIT_DStream_t *bitD,
                                                U32 nbBits) {
  U32 const regMask = sizeof(bitD->bitContainer) * 8 - 1;
  return (bitD->bitContainer << (bitD->bitsConsumed & regMask)) >>
         (((regMask + 1) - nbBits) & regMask);
}

DEVICE_INLINE void BIT_skipBits(BIT_DStream_t *bitD, U32 nbBits) {
  bitD->bitsConsumed += nbBits;
}

DEVICE_INLINE BitContainerType BIT_readBits(BIT_DStream_t *bitD,
                                            unsigned nbBits) {
  BitContainerType const value = BIT_lookBits(bitD, nbBits);
  BIT_skipBits(bitD, nbBits);
  return value;
}

DEVICE_INLINE BitContainerType BIT_readBitsFast(BIT_DStream_t *bitD,
                                                unsigned nbBits) {
  BitContainerType const value = BIT_lookBitsFast(bitD, nbBits);
  BIT_skipBits(bitD, nbBits);
  return value;
}

DEVICE_INLINE BIT_DStream_status
BIT_reloadDStream_internal(BIT_DStream_t *bitD) {
  bitD->ptr -= bitD->bitsConsumed >> 3;
  bitD->bitsConsumed &= 7;
  bitD->bitContainer = MEM_readLEST(bitD->ptr);
  return BIT_DStream_unfinished;
}

DEVICE_INLINE BIT_DStream_status BIT_reloadDStream(BIT_DStream_t *bitD) {

  if ((__builtin_expect((bitD->bitsConsumed > (sizeof(bitD->bitContainer) * 8)),
                        0))) {
    static const BitContainerType zeroFilled = 0;
    bitD->ptr = (const char *)&zeroFilled;

    return BIT_DStream_overflow;
  }

  if (bitD->ptr >= bitD->limitPtr) {
    return BIT_reloadDStream_internal(bitD);
  }
  if (bitD->ptr == bitD->start) {

    if (bitD->bitsConsumed < sizeof(bitD->bitContainer) * 8)
      return BIT_DStream_endOfBuffer;
    return BIT_DStream_completed;
  }

  {
    U32 nbBytes = bitD->bitsConsumed >> 3;
    BIT_DStream_status result = BIT_DStream_unfinished;
    if (bitD->ptr - nbBytes < bitD->start) {
      nbBytes = (U32)(bitD->ptr - bitD->start);
      result = BIT_DStream_endOfBuffer;
    }
    bitD->ptr -= nbBytes;
    bitD->bitsConsumed -= nbBytes * 8;
    bitD->bitContainer = MEM_readLEST(bitD->ptr);
    return result;
  }
}

DEVICE_INLINE BIT_DStream_status BIT_reloadDStreamFast(BIT_DStream_t *bitD) {
  if ((__builtin_expect((bitD->ptr < bitD->limitPtr), 0)))
    return BIT_DStream_overflow;
  return BIT_reloadDStream_internal(bitD);
}

DEVICE_INLINE unsigned BIT_endOfDStream(const BIT_DStream_t *DStream) {
  return ((DStream->ptr == DStream->start) &&
          (DStream->bitsConsumed == sizeof(DStream->bitContainer) * 8));
}

} // namespace zstd
} // namespace hipcomp
