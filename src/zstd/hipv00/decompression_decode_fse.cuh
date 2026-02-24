/* ******************************************************************
 * FSE : Finite State Entropy decoder
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 *  You can contact the author at :
 *  - FSE source repository : https://github.com/Cyan4973/FiniteStateEntropy
 *  - Public forum : https://groups.google.com/forum/#!forum/lz4c
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 ****************************************************************** */

// MIT License
//
// Modifications Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights
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

#include "bitstream.cuh"
#include "common.cuh"

namespace hipcomp {
namespace zstd {

/* ***************************************************************
 *  Constants
 *****************************************************************/
#define FSE_MAX_TABLELOG (FSE_MAX_MEMORY_USAGE - 2)
#define FSE_MAX_TABLESIZE (1U << FSE_MAX_TABLELOG)
#define FSE_MAXTABLESIZE_MASK (FSE_MAX_TABLESIZE - 1)
#define FSE_DEFAULT_TABLELOG (FSE_DEFAULT_MEMORY_USAGE - 2)
#define FSE_MIN_TABLELOG 5

#define FSE_TABLELOG_ABSOLUTE_MAX 15
#if FSE_MAX_TABLELOG > FSE_TABLELOG_ABSOLUTE_MAX
#error "FSE_MAX_TABLELOG > FSE_TABLELOG_ABSOLUTE_MAX is not supported"
#endif

#define FSE_TABLESTEP(tableSize) (((tableSize) >> 1) + ((tableSize) >> 3) + 3)

// FSE

typedef unsigned FSE_CTable;

typedef unsigned FSE_DTable;

typedef struct {
  ptrdiff_t value;
  const void *stateTable;
  const void *symbolTT;
  unsigned stateLog;
} FSE_CState_t;

typedef struct {
  size_t state;
  const void *table;
} FSE_DState_t;

typedef struct {
  int deltaFindState;
  U32 deltaNbBits;
} FSE_symbolCompressionTransform;

DEVICE_INLINE void FSE_initCState(FSE_CState_t *statePtr,
                                  const FSE_CTable *ct) {
  const void *ptr = ct;
  const U16 *u16ptr = (const U16 *)ptr;
  const U32 tableLog = MEM_read16(ptr);
  statePtr->value = (ptrdiff_t)1 << tableLog;
  statePtr->stateTable = u16ptr + 2;
  statePtr->symbolTT = ct + 1 + (tableLog ? (1 << (tableLog - 1)) : 1);
  statePtr->stateLog = tableLog;
}

DEVICE_INLINE void FSE_initCState2(FSE_CState_t *statePtr, const FSE_CTable *ct,
                                   U32 symbol) {
  FSE_initCState(statePtr, ct);
  {
    const FSE_symbolCompressionTransform symbolTT =
        ((const FSE_symbolCompressionTransform *)(statePtr->symbolTT))[symbol];
    const U16 *stateTable = (const U16 *)(statePtr->stateTable);
    U32 nbBitsOut = (U32)((symbolTT.deltaNbBits + (1 << 15)) >> 16);
    statePtr->value = (nbBitsOut << 16) - symbolTT.deltaNbBits;
    statePtr->value =
        stateTable[(statePtr->value >> nbBitsOut) + symbolTT.deltaFindState];
  }
}

DEVICE_INLINE void FSE_encodeSymbol(BIT_CStream_t *bitC, FSE_CState_t *statePtr,
                                    unsigned symbol) {
  FSE_symbolCompressionTransform const symbolTT =
      ((const FSE_symbolCompressionTransform *)(statePtr->symbolTT))[symbol];
  const U16 *const stateTable = (const U16 *)(statePtr->stateTable);
  U32 const nbBitsOut = (U32)((statePtr->value + symbolTT.deltaNbBits) >> 16);
  BIT_addBits(bitC, (BitContainerType)statePtr->value, nbBitsOut);
  statePtr->value =
      stateTable[(statePtr->value >> nbBitsOut) + symbolTT.deltaFindState];
}

DEVICE_INLINE void FSE_flushCState(BIT_CStream_t *bitC,
                                   const FSE_CState_t *statePtr) {
  BIT_addBits(bitC, (BitContainerType)statePtr->value, statePtr->stateLog);
  BIT_flushBits(bitC);
}

DEVICE_INLINE U32 FSE_getMaxNbBits(const void *symbolTTPtr, U32 symbolValue) {
  const FSE_symbolCompressionTransform *symbolTT =
      (const FSE_symbolCompressionTransform *)symbolTTPtr;
  return (symbolTT[symbolValue].deltaNbBits + ((1 << 16) - 1)) >> 16;
}

DEVICE_INLINE U32 FSE_bitCost(const void *symbolTTPtr, U32 tableLog,
                              U32 symbolValue, U32 accuracyLog) {
  const FSE_symbolCompressionTransform *symbolTT =
      (const FSE_symbolCompressionTransform *)symbolTTPtr;
  U32 const minNbBits = symbolTT[symbolValue].deltaNbBits >> 16;
  U32 const threshold = (minNbBits + 1) << 16;
  {
    U32 const tableSize = 1 << tableLog;
    U32 const deltaFromThreshold =
        threshold - (symbolTT[symbolValue].deltaNbBits + tableSize);
    U32 const normalizedDeltaFromThreshold =
        (deltaFromThreshold << accuracyLog) >> tableLog;
    U32 const bitMultiplier = 1 << accuracyLog;
    return (minNbBits + 1) * bitMultiplier - normalizedDeltaFromThreshold;
  }
}

typedef struct {
  U16 tableLog;
  U16 fastMode;
} FSE_DTableHeader;

typedef struct {
  unsigned short newState;
  unsigned char symbol;
  unsigned char nbBits;
} FSE_decode_t;

DEVICE_INLINE void FSE_initDState(FSE_DState_t *DStatePtr, BIT_DStream_t *bitD,
                                  const FSE_DTable *dt) {
  const void *ptr = dt;
  const FSE_DTableHeader *const DTableH = (const FSE_DTableHeader *)ptr;
  DStatePtr->state = BIT_readBits(bitD, DTableH->tableLog);
  BIT_reloadDStream(bitD);
  DStatePtr->table = dt + 1;
}

DEVICE_INLINE BYTE FSE_peekSymbol(const FSE_DState_t *DStatePtr) {
  FSE_decode_t const DInfo =
      ((const FSE_decode_t *)(DStatePtr->table))[DStatePtr->state];
  return DInfo.symbol;
}

DEVICE_INLINE void FSE_updateState(FSE_DState_t *DStatePtr,
                                   BIT_DStream_t *bitD) {
  FSE_decode_t const DInfo =
      ((const FSE_decode_t *)(DStatePtr->table))[DStatePtr->state];
  U32 const nbBits = DInfo.nbBits;
  size_t const lowBits = BIT_readBits(bitD, nbBits);
  DStatePtr->state = DInfo.newState + lowBits;
}

DEVICE_INLINE BYTE FSE_decodeSymbol(FSE_DState_t *DStatePtr,
                                    BIT_DStream_t *bitD) {
  FSE_decode_t const DInfo =
      ((const FSE_decode_t *)(DStatePtr->table))[DStatePtr->state];
  U32 const nbBits = DInfo.nbBits;
  BYTE const symbol = DInfo.symbol;
  size_t const lowBits = BIT_readBits(bitD, nbBits);

  DStatePtr->state = DInfo.newState + lowBits;
  return symbol;
}

DEVICE_INLINE BYTE FSE_decodeSymbolFast(FSE_DState_t *DStatePtr,
                                        BIT_DStream_t *bitD) {
  FSE_decode_t const DInfo =
      ((const FSE_decode_t *)(DStatePtr->table))[DStatePtr->state];
  U32 const nbBits = DInfo.nbBits;
  BYTE const symbol = DInfo.symbol;
  size_t const lowBits = BIT_readBitsFast(bitD, nbBits);

  DStatePtr->state = DInfo.newState + lowBits;
  return symbol;
}

DEVICE_INLINE unsigned FSE_endOfDState(const FSE_DState_t *DStatePtr) {
  return DStatePtr->state == 0;
}

static DEVICE_INLINE size_t FSE_buildDTable_internal(
    FSE_DTable *dt, const short *normalizedCounter, unsigned maxSymbolValue,
    unsigned tableLog, void *workSpace, size_t wkspSize) {
  void *const tdPtr = dt + 1;
  FSE_decode_t *const tableDecode = (FSE_decode_t *)(tdPtr);
  U16 *symbolNext = (U16 *)workSpace;
  BYTE *spread = (BYTE *)(symbolNext + maxSymbolValue + 1);

  U32 const maxSV1 = maxSymbolValue + 1;
  U32 const tableSize = 1 << tableLog;
  U32 highThreshold = tableSize - 1;

  if ((sizeof(short) * (maxSymbolValue + 1) + (1ULL << tableLog) + 8) >
      wkspSize)
    return ((size_t)-ZSTD_error_maxSymbolValue_tooLarge);
  if (maxSymbolValue > 255)
    return ((size_t)-ZSTD_error_maxSymbolValue_tooLarge);
  if (tableLog > (14 - 2))
    return ((size_t)-ZSTD_error_tableLog_tooLarge);

  {
    FSE_DTableHeader DTableH;
    DTableH.tableLog = (U16)tableLog;
    DTableH.fastMode = 1;
    {
      S16 const largeLimit = (S16)(1 << (tableLog - 1));
      U32 s;
      for (s = 0; s < maxSV1; s++) {
        if (normalizedCounter[s] == -1) {
          tableDecode[highThreshold--].symbol = (BYTE)s;
          symbolNext[s] = 1;
        } else {
          if (normalizedCounter[s] >= largeLimit)
            DTableH.fastMode = 0;
          symbolNext[s] = (U16)normalizedCounter[s];
        }
      }
    }
    __builtin_memcpy((dt), (&DTableH), (sizeof(DTableH)));
  }

  if (highThreshold == tableSize - 1) {
    size_t const tableMask = tableSize - 1;
    size_t const step = (((tableSize) >> 1) + ((tableSize) >> 3) + 3);

    {
      U64 const add = 0x0101010101010101ull;
      size_t pos = 0;
      U64 sv = 0;
      U32 s;
      for (s = 0; s < maxSV1; ++s, sv += add) {
        int i;
        int const n = normalizedCounter[s];
        MEM_write64(spread + pos, sv);
        for (i = 8; i < n; i += 8) {
          MEM_write64(spread + pos + i, sv);
        }
        pos += (size_t)n;
      }
    }

    {
      size_t position = 0;
      size_t s;
      size_t const unroll = 2;
      for (s = 0; s < (size_t)tableSize; s += unroll) {
        size_t u;
        for (u = 0; u < unroll; ++u) {
          size_t const uPosition = (position + (u * step)) & tableMask;
          tableDecode[uPosition].symbol = spread[s + u];
        }
        position = (position + (unroll * step)) & tableMask;
      }
    }
  } else {
    U32 const tableMask = tableSize - 1;
    U32 const step = (((tableSize) >> 1) + ((tableSize) >> 3) + 3);
    U32 s, position = 0;
    for (s = 0; s < maxSV1; s++) {
      int i;
      for (i = 0; i < normalizedCounter[s]; i++) {
        tableDecode[position].symbol = (BYTE)s;
        position = (position + step) & tableMask;
        while (position > highThreshold)
          position = (position + step) & tableMask;
      }
    }
    if (position != 0)
      return ((size_t)-ZSTD_error_GENERIC);
  }

  {
    U32 u;
    for (u = 0; u < tableSize; u++) {
      BYTE const symbol = (BYTE)(tableDecode[u].symbol);
      U32 const nextState = symbolNext[symbol]++;
      tableDecode[u].nbBits = (BYTE)(tableLog - ZSTD_highbit32(nextState));
      tableDecode[u].newState =
          (U16)((nextState << tableDecode[u].nbBits) - tableSize);
    }
  }

  return 0;
}

DEVICE_INLINE size_t FSE_buildDTable_wksp(FSE_DTable *dt,
                                          const short *normalizedCounter,
                                          unsigned maxSymbolValue,
                                          unsigned tableLog, void *workSpace,
                                          size_t wkspSize) {
  return FSE_buildDTable_internal(dt, normalizedCounter, maxSymbolValue,
                                  tableLog, workSpace, wkspSize);
}

DEVICE_INLINE size_t FSE_decompress_usingDTable_generic(
    void *dst, size_t maxDstSize, const void *cSrc, size_t cSrcSize,
    const FSE_DTable *dt, const unsigned fast) {
  BYTE *const ostart = (BYTE *)dst;
  BYTE *op = ostart;
  BYTE *const omax = op + maxDstSize;
  BYTE *const olimit = omax - 3;

  BIT_DStream_t bitD;
  FSE_DState_t state1;
  FSE_DState_t state2;

  do {
    size_t const _var_err__ = BIT_initDStream(&bitD, cSrc, cSrcSize);
    do {
      if (ERR_isError(_var_err__))
        return _var_err__;
    } while (0);
  } while (0);

  FSE_initDState(&state1, &bitD, dt);
  FSE_initDState(&state2, &bitD, dt);

  do {
    if (BIT_reloadDStream(&bitD) == BIT_DStream_overflow) {
      do {
      } while (0);
      do {
        if (0) {
          _force_has_format_string("");
        }
      } while (0);
      do {
      } while (0);
      do {
      } while (0);
      return ((size_t)-ZSTD_error_corruption_detected);
    }
  } while (0);

  for (; (BIT_reloadDStream(&bitD) == BIT_DStream_unfinished) & (op < olimit);
       op += 4) {
    op[0] = fast ? FSE_decodeSymbolFast(&state1, &bitD)
                 : FSE_decodeSymbol(&state1, &bitD);

    if ((14 - 2) * 2 + 7 > sizeof(bitD.bitContainer) * 8)
      BIT_reloadDStream(&bitD);

    op[1] = fast ? FSE_decodeSymbolFast(&state2, &bitD)
                 : FSE_decodeSymbol(&state2, &bitD);

    if ((14 - 2) * 4 + 7 > sizeof(bitD.bitContainer) * 8) {
      if (BIT_reloadDStream(&bitD) > BIT_DStream_unfinished) {
        op += 2;
        break;
      }
    }

    op[2] = fast ? FSE_decodeSymbolFast(&state1, &bitD)
                 : FSE_decodeSymbol(&state1, &bitD);

    if ((14 - 2) * 2 + 7 > sizeof(bitD.bitContainer) * 8)
      BIT_reloadDStream(&bitD);

    op[3] = fast ? FSE_decodeSymbolFast(&state2, &bitD)
                 : FSE_decodeSymbol(&state2, &bitD);
  }

  while (1) {
    if (op > (omax - 2))
      return ((size_t)-ZSTD_error_dstSize_tooSmall);
    *op++ = fast ? FSE_decodeSymbolFast(&state1, &bitD)
                 : FSE_decodeSymbol(&state1, &bitD);
    if (BIT_reloadDStream(&bitD) == BIT_DStream_overflow) {
      *op++ = fast ? FSE_decodeSymbolFast(&state2, &bitD)
                   : FSE_decodeSymbol(&state2, &bitD);
      break;
    }

    if (op > (omax - 2))
      return ((size_t)-ZSTD_error_dstSize_tooSmall);
    *op++ = fast ? FSE_decodeSymbolFast(&state2, &bitD)
                 : FSE_decodeSymbol(&state2, &bitD);
    if (BIT_reloadDStream(&bitD) == BIT_DStream_overflow) {
      *op++ = fast ? FSE_decodeSymbolFast(&state1, &bitD)
                   : FSE_decodeSymbol(&state1, &bitD);
      break;
    }
  }
  return (size_t)(op - ostart);
}

typedef struct {
  short ncount[255 + 1];
} FSE_DecompressWksp;

/*===   Error Management   ===*/
// DEVICE_INLINE unsigned FSE_isError(size_t code) { return ERR_isError(code); }

/*-**************************************************************
 *  FSE NCount encoding-decoding
 ****************************************************************/

// forward declare
DEVICE_INLINE size_t FSE_readNCount(short *normalizedCounter,
                                    unsigned *maxSVPtr, unsigned *tableLogPtr,
                                    const void *headerBuffer, size_t hbSize);

DEVICE_INLINE
size_t FSE_readNCount_body(short *normalizedCounter, unsigned *maxSVPtr,
                           unsigned *tableLogPtr, const void *headerBuffer,
                           size_t hbSize) {
  const BYTE *const istart = (const BYTE *)headerBuffer;
  const BYTE *const iend = istart + hbSize;
  const BYTE *ip = istart;
  int nbBits;
  int remaining;
  int threshold;
  U32 bitStream;
  int bitCount;
  unsigned charnum = 0;
  unsigned const maxSV1 = *maxSVPtr + 1;
  int previous0 = 0;

  if (hbSize < 8) {
    /* This function only works when hbSize >= 8 */
    char buffer[8] = {0};
    ZSTD_memcpy(buffer, headerBuffer, hbSize);
    {
      size_t const countSize = FSE_readNCount(
          normalizedCounter, maxSVPtr, tableLogPtr, buffer, sizeof(buffer));
      if (FSE_isError(countSize))
        return countSize;
      if (countSize > hbSize)
        return ERROR(corruption_detected);
      return countSize;
    }
  }
  assert(hbSize >= 8);

  /* init */
  ZSTD_memset(
      normalizedCounter, 0,
      (*maxSVPtr + 1) *
          sizeof(normalizedCounter[0])); /* all symbols not present in NCount
                                            have a frequency of 0 */
  bitStream = MEM_readLE32(ip);
  nbBits = (bitStream & 0xF) + FSE_MIN_TABLELOG; /* extract tableLog */
  if (nbBits > FSE_TABLELOG_ABSOLUTE_MAX)
    return ERROR(tableLog_tooLarge);
  bitStream >>= 4;
  bitCount = 4;
  *tableLogPtr = nbBits;
  remaining = (1 << nbBits) + 1;
  threshold = 1 << nbBits;
  nbBits++;

  for (;;) {
    if (previous0) {
      /* Count the number of repeats. Each time the
       * 2-bit repeat code is 0b11 there is another
       * repeat.
       * Avoid UB by setting the high bit to 1.
       */
      int repeats = ZSTD_countTrailingZeros32(~bitStream | 0x80000000) >> 1;
      while (repeats >= 12) {
        charnum += 3 * 12;
        if (LIKELY(ip <= iend - 7)) {
          ip += 3;
        } else {
          bitCount -= (int)(8 * (iend - 7 - ip));
          bitCount &= 31;
          ip = iend - 4;
        }
        bitStream = MEM_readLE32(ip) >> bitCount;
        repeats = ZSTD_countTrailingZeros32(~bitStream | 0x80000000) >> 1;
      }
      charnum += 3 * repeats;
      bitStream >>= 2 * repeats;
      bitCount += 2 * repeats;

      /* Add the final repeat which isn't 0b11. */
      assert((bitStream & 3) < 3);
      charnum += bitStream & 3;
      bitCount += 2;

      /* This is an error, but break and return an error
       * at the end, because returning out of a loop makes
       * it harder for the compiler to optimize.
       */
      if (charnum >= maxSV1)
        break;

      /* We don't need to set the normalized count to 0
       * because we already memset the whole buffer to 0.
       */

      if (LIKELY(ip <= iend - 7) || (ip + (bitCount >> 3) <= iend - 4)) {
        assert((bitCount >> 3) <= 3); /* For first condition to work */
        ip += bitCount >> 3;
        bitCount &= 7;
      } else {
        bitCount -= (int)(8 * (iend - 4 - ip));
        bitCount &= 31;
        ip = iend - 4;
      }
      bitStream = MEM_readLE32(ip) >> bitCount;
    }
    {
      int const max = (2 * threshold - 1) - remaining;
      int count;

      if ((bitStream & (threshold - 1)) < (U32)max) {
        count = bitStream & (threshold - 1);
        bitCount += nbBits - 1;
      } else {
        count = bitStream & (2 * threshold - 1);
        if (count >= threshold)
          count -= max;
        bitCount += nbBits;
      }

      count--; /* extra accuracy */
      /* When it matters (small blocks), this is a
       * predictable branch, because we don't use -1.
       */
      if (count >= 0) {
        remaining -= count;
      } else {
        assert(count == -1);
        remaining += count;
      }
      normalizedCounter[charnum++] = (short)count;
      previous0 = !count;

      assert(threshold > 1);
      if (remaining < threshold) {
        /* This branch can be folded into the
         * threshold update condition because we
         * know that threshold > 1.
         */
        if (remaining <= 1)
          break;
        nbBits = ZSTD_highbit32(remaining) + 1;
        threshold = 1 << (nbBits - 1);
      }
      if (charnum >= maxSV1)
        break;

      if (LIKELY(ip <= iend - 7) || (ip + (bitCount >> 3) <= iend - 4)) {
        ip += bitCount >> 3;
        bitCount &= 7;
      } else {
        bitCount -= (int)(8 * (iend - 4 - ip));
        bitCount &= 31;
        ip = iend - 4;
      }
      bitStream = MEM_readLE32(ip) >> bitCount;
    }
  }
  if (remaining != 1)
    return ERROR(corruption_detected);
  /* Only possible when there are too many zeros. */
  if (charnum > maxSV1)
    return ERROR(maxSymbolValue_tooSmall);
  if (bitCount > 32)
    return ERROR(corruption_detected);
  *maxSVPtr = charnum - 1;

  ip += (bitCount + 7) >> 3;
  return ip - istart;
}

/* Avoids the FORCE_INLINE of the _body() function. */
DEVICE_INLINE size_t FSE_readNCount_body_default(short *normalizedCounter,
                                                 unsigned *maxSVPtr,
                                                 unsigned *tableLogPtr,
                                                 const void *headerBuffer,
                                                 size_t hbSize) {
  return FSE_readNCount_body(normalizedCounter, maxSVPtr, tableLogPtr,
                             headerBuffer, hbSize);
}

#if DYNAMIC_BMI2
DEVICE_INLINE BMI2_TARGET_ATTRIBUTE static size_t
FSE_readNCount_body_bmi2(short *normalizedCounter, unsigned *maxSVPtr,
                         unsigned *tableLogPtr, const void *headerBuffer,
                         size_t hbSize) {
  return FSE_readNCount_body(normalizedCounter, maxSVPtr, tableLogPtr,
                             headerBuffer, hbSize);
}
#endif

DEVICE_INLINE size_t FSE_readNCount_bmi2(short *normalizedCounter,
                                         unsigned *maxSVPtr,
                                         unsigned *tableLogPtr,
                                         const void *headerBuffer,
                                         size_t hbSize, int bmi2) {
#if DYNAMIC_BMI2
  if (bmi2) {
    return FSE_readNCount_body_bmi2(normalizedCounter, maxSVPtr, tableLogPtr,
                                    headerBuffer, hbSize);
  }
#endif
  (void)bmi2;
  return FSE_readNCount_body_default(normalizedCounter, maxSVPtr, tableLogPtr,
                                     headerBuffer, hbSize);
}

DEVICE_INLINE size_t FSE_readNCount(short *normalizedCounter,
                                    unsigned *maxSVPtr, unsigned *tableLogPtr,
                                    const void *headerBuffer, size_t hbSize) {
  return FSE_readNCount_bmi2(normalizedCounter, maxSVPtr, tableLogPtr,
                             headerBuffer, hbSize, /* bmi2 */ 0);
}

DEVICE_INLINE size_t FSE_decompress_wksp_body(void *dst, size_t dstCapacity,
                                              const void *cSrc, size_t cSrcSize,
                                              unsigned maxLog, void *workSpace,
                                              size_t wkspSize, int bmi2) {
  const BYTE *const istart = (const BYTE *)cSrc;
  const BYTE *ip = istart;
  unsigned tableLog;
  unsigned maxSymbolValue = 255;
  FSE_DecompressWksp *const wksp = (FSE_DecompressWksp *)workSpace;
  size_t const dtablePos = sizeof(FSE_DecompressWksp) / sizeof(FSE_DTable);
  FSE_DTable *const dtable = (FSE_DTable *)workSpace + dtablePos;

  (void)sizeof(char[((255 + 1) % 2 == 0) ? 1 : -1]);
  if (wkspSize < sizeof(*wksp))
    return ((size_t)-ZSTD_error_GENERIC);

  (void)sizeof(
      char[(sizeof(FSE_DecompressWksp) % sizeof(FSE_DTable) == 0) ? 1 : -1]);

  {
    size_t const NCountLength = FSE_readNCount_bmi2(
        wksp->ncount, &maxSymbolValue, &tableLog, istart, cSrcSize, bmi2);
    if (ERR_isError(NCountLength))
      return NCountLength;
    if (tableLog > maxLog)
      return ((size_t)-ZSTD_error_tableLog_tooLarge);
    ip += NCountLength;
    cSrcSize -= NCountLength;
  }

  if ((((1 + (1 << (tableLog))) + 1 +
        (((sizeof(short) * (maxSymbolValue + 1) + (1ULL << tableLog) + 8) +
          sizeof(unsigned) - 1) /
         sizeof(unsigned)) +
        (255 + 1) / 2 + 1) *
       sizeof(unsigned)) > wkspSize)
    return ((size_t)-ZSTD_error_tableLog_tooLarge);
  workSpace = (BYTE *)workSpace + sizeof(*wksp) +
              ((1 + (1 << (tableLog))) * sizeof(FSE_DTable));
  wkspSize -= sizeof(*wksp) + ((1 + (1 << (tableLog))) * sizeof(FSE_DTable));

  do {
    size_t const _var_err__ = FSE_buildDTable_internal(
        dtable, wksp->ncount, maxSymbolValue, tableLog, workSpace, wkspSize);
    do {
      if (ERR_isError(_var_err__))
        return _var_err__;
    } while (0);
  } while (0);

  {
    const void *ptr = dtable;
    const FSE_DTableHeader *DTableH = (const FSE_DTableHeader *)ptr;
    const U32 fastMode = DTableH->fastMode;

    if (fastMode)
      return FSE_decompress_usingDTable_generic(dst, dstCapacity, ip, cSrcSize,
                                                dtable, 1);
    return FSE_decompress_usingDTable_generic(dst, dstCapacity, ip, cSrcSize,
                                              dtable, 0);
  }
}

DEVICE_INLINE size_t FSE_decompress_wksp_body_default(
    void *dst, size_t dstCapacity, const void *cSrc, size_t cSrcSize,
    unsigned maxLog, void *workSpace, size_t wkspSize) {
  return FSE_decompress_wksp_body(dst, dstCapacity, cSrc, cSrcSize, maxLog,
                                  workSpace, wkspSize, 0);
}

DEVICE_INLINE size_t FSE_decompress_wksp_body_bmi2(
    void *dst, size_t dstCapacity, const void *cSrc, size_t cSrcSize,
    unsigned maxLog, void *workSpace, size_t wkspSize) {
  return FSE_decompress_wksp_body(dst, dstCapacity, cSrc, cSrcSize, maxLog,
                                  workSpace, wkspSize, 1);
}

DEVICE_INLINE size_t FSE_decompress_wksp_bmi2(void *dst, size_t dstCapacity,
                                              const void *cSrc, size_t cSrcSize,
                                              unsigned maxLog, void *workSpace,
                                              size_t wkspSize, int bmi2) {

  if (bmi2) {
    return FSE_decompress_wksp_body_bmi2(dst, dstCapacity, cSrc, cSrcSize,
                                         maxLog, workSpace, wkspSize);
  }

  (void)bmi2;
  return FSE_decompress_wksp_body_default(dst, dstCapacity, cSrc, cSrcSize,
                                          maxLog, workSpace, wkspSize);
}

} // namespace zstd
} // namespace hipcomp
