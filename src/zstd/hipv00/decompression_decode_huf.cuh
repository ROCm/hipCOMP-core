/* ******************************************************************
 * huff0 huffman decoder,
 * part of Finite State Entropy library
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 *  You can contact the author at :
 *  - FSE+HUF source repository : https://github.com/Cyan4973/FiniteStateEntropy
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 ****************************************************************** */

/* ******************************************************************
 * Common functions of New Generation Entropy library
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 *  You can contact the author at :
 *  - FSE+HUF source repository : https://github.com/Cyan4973/FiniteStateEntropy
 *  - Public forum : https://groups.google.com/forum/#!forum/lz4c
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 ****************************************************************** */

/* ******************************************************************
 * FSE : Finite State Entropy codec
 * Public Prototypes declaration
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

// #define HUF_WORKSPACE_SIZE ((8 << 10) + 512 /* sorting scratch space */)
// #define HUF_WORKSPACE_SIZE_U64 (HUF_WORKSPACE_SIZE / sizeof(U64))

// /* *** Constants *** */
#define HUF_TABLELOG_MAX                                                       \
  12 /* max runtime value of tableLog (due to static allocation); can be       \
        modified up to HUF_TABLELOG_ABSOLUTEMAX */
#define HUF_TABLELOG_DEFAULT                                                   \
  11 /* default tableLog value when none specified                             \
      */
#define HUF_SYMBOLVALUE_MAX 255

/* **************************************************************
 *  Constants
 ****************************************************************/

#define HUF_DECODER_FAST_TABLELOG 11

/* **************************************************************
 *  Macros
 ****************************************************************/

#ifdef HUF_DISABLE_FAST_DECODE
#define HUF_ENABLE_FAST_DECODE 0
#else
#define HUF_ENABLE_FAST_DECODE 1
#endif

/* These two optional macros force the use one way or another of the two
 * Huffman decompression implementations. You can't force in both directions
 * at the same time.
 */
#if defined(HUF_FORCE_DECOMPRESS_X1) && defined(HUF_FORCE_DECOMPRESS_X2)
#error "Cannot force the use of the X1 and X2 decoders at the same time!"
#endif

// #define HUF_TABLELOG_ABSOLUTEMAX  12  /* absolute limit of HUF_MAX_TABLELOG.
// Beyond that value, code does not work */ #if (HUF_TABLELOG_MAX >
// HUF_TABLELOG_ABSOLUTEMAX) #  error "HUF_TABLELOG_MAX is too large !" #endif

/*!FSE_MAX_SYMBOL_VALUE :
 *  Maximum symbol value authorized.
 *  Required for proper stack allocation */
#ifndef FSE_MAX_SYMBOL_VALUE
#define FSE_MAX_SYMBOL_VALUE 255
#endif

#define FSE_DTABLE_SIZE_U32(maxTableLog) (1 + (1 << (maxTableLog)))
#define FSE_BUILD_DTABLE_WKSP_SIZE(maxTableLog, maxSymbolValue)                \
  (sizeof(short) * (maxSymbolValue + 1) + (1ULL << maxTableLog) + 8)
#define FSE_BUILD_DTABLE_WKSP_SIZE_U32(maxTableLog, maxSymbolValue)            \
  ((FSE_BUILD_DTABLE_WKSP_SIZE(maxTableLog, maxSymbolValue) +                  \
    sizeof(unsigned) - 1) /                                                    \
   sizeof(unsigned))
#define FSE_DECOMPRESS_WKSP_SIZE_U32(maxTableLog, maxSymbolValue)              \
  (FSE_DTABLE_SIZE_U32(maxTableLog) + 1 +                                      \
   FSE_BUILD_DTABLE_WKSP_SIZE_U32(maxTableLog, maxSymbolValue) +               \
   (FSE_MAX_SYMBOL_VALUE + 1) / 2 + 1)
#define FSE_DECOMPRESS_WKSP_SIZE(maxTableLog, maxSymbolValue)                  \
  (FSE_DECOMPRESS_WKSP_SIZE_U32(maxTableLog, maxSymbolValue) * sizeof(unsigned))

#define HUF_READ_STATS_WORKSPACE_SIZE_U32                                      \
  FSE_DECOMPRESS_WKSP_SIZE_U32(6, HUF_TABLELOG_MAX - 1)
#define HUF_READ_STATS_WORKSPACE_SIZE                                          \
  (HUF_READ_STATS_WORKSPACE_SIZE_U32 * sizeof(unsigned))

/* Calls X(N) for each stream 0, 1, 2, 3. */
#define HUF_4X_FOR_EACH_STREAM(X)                                              \
  do {                                                                         \
    X(0);                                                                      \
    X(1);                                                                      \
    X(2);                                                                      \
    X(3);                                                                      \
  } while (0)

/* Calls X(N, var) for each stream 0, 1, 2, 3. */
#define HUF_4X_FOR_EACH_STREAM_WITH_VAR(X, var)                                \
  do {                                                                         \
    X(0, (var));                                                               \
    X(1, (var));                                                               \
    X(2, (var));                                                               \
    X(3, (var));                                                               \
  } while (0)

// forward declare
DEVICE_INLINE size_t HUF_readStats_wksp(BYTE *huffWeight, size_t hwSize,
                                        U32 *rankStats, U32 *nbSymbolsPtr,
                                        U32 *tableLogPtr, const void *src,
                                        size_t srcSize, void *workSpace,
                                        size_t wkspSize, int flags);

/*! HUF_readStats() :
    Read compact Huffman tree, saved by HUF_writeCTable().
    `huffWeight` is destination buffer.
    `rankStats` is assumed to be a table of at least HUF_TABLELOG_MAX U32.
    @return : size read from `src` , or an error Code .
    Note : Needed by HUF_readCTable() and HUF_readDTableX?() .
*/
DEVICE_INLINE size_t HUF_readStats(BYTE *huffWeight, size_t hwSize,
                                   U32 *rankStats, U32 *nbSymbolsPtr,
                                   U32 *tableLogPtr, const void *src,
                                   size_t srcSize) {
  U32 wksp[HUF_READ_STATS_WORKSPACE_SIZE_U32];
  return HUF_readStats_wksp(huffWeight, hwSize, rankStats, nbSymbolsPtr,
                            tableLogPtr, src, srcSize, wksp, sizeof(wksp),
                            /* flags */ 0);
}

DEVICE_INLINE size_t HUF_readStats_body(BYTE *huffWeight, size_t hwSize,
                                        U32 *rankStats, U32 *nbSymbolsPtr,
                                        U32 *tableLogPtr, const void *src,
                                        size_t srcSize, void *workSpace,
                                        size_t wkspSize, int bmi2) {
  U32 weightTotal;
  const BYTE *ip = (const BYTE *)src;
  size_t iSize;
  size_t oSize;

  if (!srcSize)
    return ERROR(srcSize_wrong);
  iSize = ip[0];
  /* ZSTD_memset(huffWeight, 0, hwSize);   */ /* is not necessary, even though
                                                 some analyzer complain ... */

  if (iSize >= 128) { /* special header */
    oSize = iSize - 127;
    iSize = ((oSize + 1) / 2);
    if (iSize + 1 > srcSize)
      return ERROR(srcSize_wrong);
    if (oSize >= hwSize)
      return ERROR(corruption_detected);
    ip += 1;
    {
      U32 n;
      for (n = 0; n < oSize; n += 2) {
        huffWeight[n] = ip[n / 2] >> 4;
        huffWeight[n + 1] = ip[n / 2] & 15;
      }
    }
  } else { /* header compressed with FSE (normal case) */
    if (iSize + 1 > srcSize)
      return ERROR(srcSize_wrong);
    /* max (hwSize-1) values decoded, as last one is implied */
    oSize = FSE_decompress_wksp_bmi2(huffWeight, hwSize - 1, ip + 1, iSize, 6,
                                     workSpace, wkspSize, bmi2);
    if (FSE_isError(oSize))
      return oSize;
  }

  /* collect weight stats */
  ZSTD_memset(rankStats, 0, (HUF_TABLELOG_MAX + 1) * sizeof(U32));
  weightTotal = 0;
  {
    U32 n;
    for (n = 0; n < oSize; n++) {
      if (huffWeight[n] > HUF_TABLELOG_MAX)
        return ERROR(corruption_detected);
      rankStats[huffWeight[n]]++;
      weightTotal += (1 << huffWeight[n]) >> 1;
    }
  }
  if (weightTotal == 0)
    return ERROR(corruption_detected);

  /* get last non-null symbol weight (implied, total must be 2^n) */
  {
    U32 const tableLog = ZSTD_highbit32(weightTotal) + 1;
    if (tableLog > HUF_TABLELOG_MAX)
      return ERROR(corruption_detected);
    *tableLogPtr = tableLog;
    /* determine last weight */
    {
      U32 const total = 1 << tableLog;
      U32 const rest = total - weightTotal;
      U32 const verif = 1 << ZSTD_highbit32(rest);
      U32 const lastWeight = ZSTD_highbit32(rest) + 1;
      if (verif != rest)
        return ERROR(
            corruption_detected); /* last value must be a clean power of 2 */
      huffWeight[oSize] = (BYTE)lastWeight;
      rankStats[lastWeight]++;
    }
  }

  /* check tree construction validity */
  if ((rankStats[1] < 2) || (rankStats[1] & 1))
    return ERROR(corruption_detected); /* by construction : at least 2 elts of
                                          rank 1, must be even */

  /* results */
  *nbSymbolsPtr = (U32)(oSize + 1);
  return iSize + 1;
}

/* Avoids the FORCE_INLINE of the _body() function. */
DEVICE_INLINE size_t
HUF_readStats_body_default(BYTE *huffWeight, size_t hwSize, U32 *rankStats,
                           U32 *nbSymbolsPtr, U32 *tableLogPtr, const void *src,
                           size_t srcSize, void *workSpace, size_t wkspSize) {
  return HUF_readStats_body(huffWeight, hwSize, rankStats, nbSymbolsPtr,
                            tableLogPtr, src, srcSize, workSpace, wkspSize, 0);
}

#if DYNAMIC_BMI2
DEVICE_INLINE BMI2_TARGET_ATTRIBUTE size_t
HUF_readStats_body_bmi2(BYTE *huffWeight, size_t hwSize, U32 *rankStats,
                        U32 *nbSymbolsPtr, U32 *tableLogPtr, const void *src,
                        size_t srcSize, void *workSpace, size_t wkspSize) {
  return HUF_readStats_body(huffWeight, hwSize, rankStats, nbSymbolsPtr,
                            tableLogPtr, src, srcSize, workSpace, wkspSize, 1);
}
#endif

DEVICE_INLINE size_t HUF_readStats_wksp(BYTE *huffWeight, size_t hwSize,
                                        U32 *rankStats, U32 *nbSymbolsPtr,
                                        U32 *tableLogPtr, const void *src,
                                        size_t srcSize, void *workSpace,
                                        size_t wkspSize, int flags) {
#if DYNAMIC_BMI2
  if (flags & HUF_flags_bmi2) {
    return HUF_readStats_body_bmi2(huffWeight, hwSize, rankStats, nbSymbolsPtr,
                                   tableLogPtr, src, srcSize, workSpace,
                                   wkspSize);
  }
#endif
  (void)flags;
  return HUF_readStats_body_default(huffWeight, hwSize, rankStats, nbSymbolsPtr,
                                    tableLogPtr, src, srcSize, workSpace,
                                    wkspSize);
}

typedef U32 HUF_DTable;

typedef enum {

  HUF_flags_bmi2 = (1 << 0),

  HUF_flags_optimalDepth = (1 << 1),

  HUF_flags_preferRepeat = (1 << 2),

  HUF_flags_suspectUncompressible = (1 << 3),

  HUF_flags_disableAsm = (1 << 4),

  HUF_flags_disableFast = (1 << 5)
} HUF_flags_e;

typedef size_t (*HUF_DecompressUsingDTableFn)(void *dst, size_t dstSize,
                                              const void *cSrc, size_t cSrcSize,
                                              const HUF_DTable *DTable);

typedef struct {
  BYTE maxTableLog;
  BYTE tableType;
  BYTE tableLog;
  BYTE reserved;
} DTableDesc;

DEVICE_INLINE DTableDesc HUF_getDTableDesc(const HUF_DTable *table) {
  DTableDesc dtd;
  __builtin_memcpy((&dtd), (table), (sizeof(dtd)));
  return dtd;
}

DEVICE_INLINE size_t HUF_initFastDStream(BYTE const *ip) {
  BYTE const lastByte = ip[7];
  size_t const bitsConsumed = lastByte ? 8 - ZSTD_highbit32(lastByte) : 0;
  size_t const value = MEM_readLEST(ip) | 1;

  return value << bitsConsumed;
}

typedef struct {
  BYTE const *ip[4];
  BYTE *op[4];
  U64 bits[4];
  void const *dt;
  BYTE const *ilowest;
  BYTE *oend;
  BYTE const *iend[4];
} HUF_DecompressFastArgs;

typedef void (*HUF_DecompressFastLoopFn)(HUF_DecompressFastArgs *);

DEVICE_INLINE size_t HUF_DecompressFastArgs_init(HUF_DecompressFastArgs *args,
                                                 void *dst, size_t dstSize,
                                                 void const *src,
                                                 size_t srcSize,
                                                 const HUF_DTable *DTable) {
  void const *dt = DTable + 1;
  U32 const dtLog = HUF_getDTableDesc(DTable).tableLog;

  const BYTE *const istart = (const BYTE *)src;

  BYTE *const oend = (BYTE *)ZSTD_maybeNullPtrAdd(dst, (ptrdiff_t)dstSize);

  if (!MEM_isLittleEndian() || MEM_32bits())
    return 0;

  if (dstSize == 0)
    return 0;

  if (srcSize < 10)
    return ((size_t)-ZSTD_error_corruption_detected);

  if (dtLog != 11)
    return 0;

  {
    size_t const length1 = MEM_readLE16(istart);
    size_t const length2 = MEM_readLE16(istart + 2);
    size_t const length3 = MEM_readLE16(istart + 4);
    size_t const length4 = srcSize - (length1 + length2 + length3 + 6);
    args->iend[0] = istart + 6;
    args->iend[1] = args->iend[0] + length1;
    args->iend[2] = args->iend[1] + length2;
    args->iend[3] = args->iend[2] + length3;

    if (length1 < 8 || length2 < 8 || length3 < 8 || length4 < 8)
      return 0;
    if (length4 > srcSize)
      return ((size_t)-ZSTD_error_corruption_detected);
  }

  args->ip[0] = args->iend[1] - sizeof(U64);
  args->ip[1] = args->iend[2] - sizeof(U64);
  args->ip[2] = args->iend[3] - sizeof(U64);
  args->ip[3] = (BYTE const *)src + srcSize - sizeof(U64);

  args->op[0] = (BYTE *)dst;
  args->op[1] = args->op[0] + (dstSize + 3) / 4;
  args->op[2] = args->op[1] + (dstSize + 3) / 4;
  args->op[3] = args->op[2] + (dstSize + 3) / 4;

  if (args->op[3] >= oend)
    return 0;
  args->bits[0] = HUF_initFastDStream(args->ip[0]);
  args->bits[1] = HUF_initFastDStream(args->ip[1]);
  args->bits[2] = HUF_initFastDStream(args->ip[2]);
  args->bits[3] = HUF_initFastDStream(args->ip[3]);

  args->ilowest = istart;

  args->oend = oend;
  args->dt = dt;

  return 1;
}

DEVICE_INLINE size_t
HUF_initRemainingDStream(BIT_DStream_t *bit, HUF_DecompressFastArgs const *args,
                         int stream, BYTE *segmentEnd) {

  if (args->op[stream] > segmentEnd)
    return ((size_t)-ZSTD_error_corruption_detected);

  if (args->ip[stream] < args->iend[stream] - 8)
    return ((size_t)-ZSTD_error_corruption_detected);

  bit->bitContainer = MEM_readLEST(args->ip[stream]);
  bit->bitsConsumed = ZSTD_countTrailingZeros64(args->bits[stream]);
  bit->start = (const char *)args->ilowest;
  bit->limitPtr = bit->start + sizeof(size_t);
  bit->ptr = (const char *)args->ip[stream];

  return 0;
}

typedef struct {
  BYTE nbBits;
  BYTE byte;
} HUF_DEltX1;

DEVICE_INLINE U64 HUF_DEltX1_set4(BYTE symbol, BYTE nbBits) {
  U64 D4;
  if (MEM_isLittleEndian()) {
    D4 = (U64)((symbol << 8) + nbBits);
  } else {
    D4 = (U64)(symbol + (nbBits << 8));
  }

  D4 *= 0x0001000100010001ULL;
  return D4;
}

DEVICE_INLINE U32 HUF_rescaleStats(BYTE *huffWeight, U32 *rankVal,
                                   U32 nbSymbols, U32 tableLog,
                                   U32 targetTableLog) {
  if (tableLog > targetTableLog)
    return tableLog;
  if (tableLog < targetTableLog) {
    U32 const scale = targetTableLog - tableLog;
    U32 s;

    for (s = 0; s < nbSymbols; ++s) {
      huffWeight[s] += (BYTE)((huffWeight[s] == 0) ? 0 : scale);
    }

    for (s = targetTableLog; s > scale; --s) {
      rankVal[s] = rankVal[s - scale];
    }
    for (s = scale; s > 0; --s) {
      rankVal[s] = 0;
    }
  }
  return targetTableLog;
}

typedef struct {
  U32 rankVal[12 + 1];
  U32 rankStart[12 + 1];
  U32 statsWksp[((1 + (1 << (6))) + 1 +
                 (((sizeof(short) * (12 - 1 + 1) + (1ULL << 6) + 8) +
                   sizeof(unsigned) - 1) /
                  sizeof(unsigned)) +
                 (255 + 1) / 2 + 1)];
  BYTE symbols[255 + 1];
  BYTE huffWeight[255 + 1];
} HUF_ReadDTableX1_Workspace;

DEVICE_INLINE size_t HUF_readDTableX1_wksp(HUF_DTable *DTable, const void *src,
                                           size_t srcSize, void *workSpace,
                                           size_t wkspSize, int flags) {
  U32 tableLog = 0;
  U32 nbSymbols = 0;
  size_t iSize;
  void *const dtPtr = DTable + 1;
  HUF_DEltX1 *const dt = (HUF_DEltX1 *)dtPtr;
  HUF_ReadDTableX1_Workspace *wksp = (HUF_ReadDTableX1_Workspace *)workSpace;

  (void)sizeof(char[(((2 << 10) + (1 << 9)) >= sizeof(*wksp)) ? 1 : -1]);
  if (sizeof(*wksp) > wkspSize)
    return ((size_t)-ZSTD_error_tableLog_tooLarge);

  (void)sizeof(char[(sizeof(DTableDesc) == sizeof(HUF_DTable)) ? 1 : -1]);

  iSize = HUF_readStats_wksp(wksp->huffWeight, 255 + 1, wksp->rankVal,
                             &nbSymbols, &tableLog, src, srcSize,
                             wksp->statsWksp, sizeof(wksp->statsWksp), flags);
  if (ERR_isError(iSize))
    return iSize;

  {
    DTableDesc dtd = HUF_getDTableDesc(DTable);
    U32 const maxTableLog = dtd.maxTableLog + 1;
    U32 const targetTableLog = ((maxTableLog) < (11) ? (maxTableLog) : (11));
    tableLog = HUF_rescaleStats(wksp->huffWeight, wksp->rankVal, nbSymbols,
                                tableLog, targetTableLog);
    if (tableLog > (U32)(dtd.maxTableLog + 1))
      return ((size_t)-ZSTD_error_tableLog_tooLarge);
    dtd.tableType = 0;
    dtd.tableLog = (BYTE)tableLog;
    __builtin_memcpy((DTable), (&dtd), (sizeof(dtd)));
  }
  {
    int n;
    U32 nextRankStart = 0;
    int const unroll = 4;
    int const nLimit = (int)nbSymbols - unroll + 1;
    for (n = 0; n < (int)tableLog + 1; n++) {
      U32 const curr = nextRankStart;
      nextRankStart += wksp->rankVal[n];
      wksp->rankStart[n] = curr;
    }
    for (n = 0; n < nLimit; n += unroll) {
      int u;
      for (u = 0; u < unroll; ++u) {
        size_t const w = wksp->huffWeight[n + u];
        wksp->symbols[wksp->rankStart[w]++] = (BYTE)(n + u);
      }
    }
    for (; n < (int)nbSymbols; ++n) {
      size_t const w = wksp->huffWeight[n];
      wksp->symbols[wksp->rankStart[w]++] = (BYTE)n;
    }
  }

  {
    U32 w;
    int symbol = wksp->rankVal[0];
    int rankStart = 0;
    for (w = 1; w < tableLog + 1; ++w) {
      int const symbolCount = wksp->rankVal[w];
      int const length = (1 << w) >> 1;
      int uStart = rankStart;
      BYTE const nbBits = (BYTE)(tableLog + 1 - w);
      int s;
      int u;
      switch (length) {
      case 1:
        for (s = 0; s < symbolCount; ++s) {
          HUF_DEltX1 D;
          D.byte = wksp->symbols[symbol + s];
          D.nbBits = nbBits;
          dt[uStart] = D;
          uStart += 1;
        }
        break;
      case 2:
        for (s = 0; s < symbolCount; ++s) {
          HUF_DEltX1 D;
          D.byte = wksp->symbols[symbol + s];
          D.nbBits = nbBits;
          dt[uStart + 0] = D;
          dt[uStart + 1] = D;
          uStart += 2;
        }
        break;
      case 4:
        for (s = 0; s < symbolCount; ++s) {
          U64 const D4 = HUF_DEltX1_set4(wksp->symbols[symbol + s], nbBits);
          MEM_write64(dt + uStart, D4);
          uStart += 4;
        }
        break;
      case 8:
        for (s = 0; s < symbolCount; ++s) {
          U64 const D4 = HUF_DEltX1_set4(wksp->symbols[symbol + s], nbBits);
          MEM_write64(dt + uStart, D4);
          MEM_write64(dt + uStart + 4, D4);
          uStart += 8;
        }
        break;
      default:
        for (s = 0; s < symbolCount; ++s) {
          U64 const D4 = HUF_DEltX1_set4(wksp->symbols[symbol + s], nbBits);
          for (u = 0; u < length; u += 16) {
            MEM_write64(dt + uStart + u + 0, D4);
            MEM_write64(dt + uStart + u + 4, D4);
            MEM_write64(dt + uStart + u + 8, D4);
            MEM_write64(dt + uStart + u + 12, D4);
          }

          uStart += length;
        }
        break;
      }
      symbol += symbolCount;
      rankStart += symbolCount * length;
    }
  }
  return iSize;
}

DEVICE_INLINE BYTE HUF_decodeSymbolX1(BIT_DStream_t *Dstream,
                                      const HUF_DEltX1 *dt, const U32 dtLog) {
  size_t const val = BIT_lookBitsFast(Dstream, dtLog);
  BYTE const c = dt[val].byte;
  BIT_skipBits(Dstream, dt[val].nbBits);
  return c;
}

#define HUF_DECODE_SYMBOLX1_0(ptr, DStreamPtr)                                 \
  do {                                                                         \
    *ptr++ = HUF_decodeSymbolX1(DStreamPtr, dt, dtLog);                        \
  } while (0)

#define HUF_DECODE_SYMBOLX1_1(ptr, DStreamPtr)                                 \
  do {                                                                         \
    if (MEM_64bits() || (HUF_TABLELOG_MAX <= 12))                              \
      HUF_DECODE_SYMBOLX1_0(ptr, DStreamPtr);                                  \
  } while (0)

#define HUF_DECODE_SYMBOLX1_2(ptr, DStreamPtr)                                 \
  do {                                                                         \
    if (MEM_64bits())                                                          \
      HUF_DECODE_SYMBOLX1_0(ptr, DStreamPtr);                                  \
  } while (0)

DEVICE_INLINE size_t HUF_decodeStreamX1(BYTE *p, BIT_DStream_t *const bitDPtr,
                                        BYTE *const pEnd,
                                        const HUF_DEltX1 *const dt,
                                        const U32 dtLog) {
  BYTE *const pStart = p;

  /* up to 4 symbols at a time */
  if ((pEnd - p) > 3) {
    while ((BIT_reloadDStream(bitDPtr) == BIT_DStream_unfinished) &
           (p < pEnd - 3)) {
      HUF_DECODE_SYMBOLX1_2(p, bitDPtr);
      HUF_DECODE_SYMBOLX1_1(p, bitDPtr);
      HUF_DECODE_SYMBOLX1_2(p, bitDPtr);
      HUF_DECODE_SYMBOLX1_0(p, bitDPtr);
    }
  } else {
    BIT_reloadDStream(bitDPtr);
  }

  /* [0-3] symbols remaining */
  if (MEM_32bits())
    while ((BIT_reloadDStream(bitDPtr) == BIT_DStream_unfinished) & (p < pEnd))
      HUF_DECODE_SYMBOLX1_0(p, bitDPtr);

  /* no more data to retrieve from bitstream, no need to reload */
  while (p < pEnd)
    HUF_DECODE_SYMBOLX1_0(p, bitDPtr);

  return (size_t)(pEnd - pStart);
}

DEVICE_INLINE size_t HUF_decompress1X1_usingDTable_internal_body(
    void *dst, size_t dstSize, const void *cSrc, size_t cSrcSize,
    const HUF_DTable *DTable) {
  BYTE *op = (BYTE *)dst;
  BYTE *const oend = (BYTE *)ZSTD_maybeNullPtrAdd(op, (ptrdiff_t)dstSize);
  const void *dtPtr = DTable + 1;
  const HUF_DEltX1 *const dt = (const HUF_DEltX1 *)dtPtr;
  BIT_DStream_t bitD;
  DTableDesc const dtd = HUF_getDTableDesc(DTable);
  U32 const dtLog = dtd.tableLog;

  CHECK_F(BIT_initDStream(&bitD, cSrc, cSrcSize));

  HUF_decodeStreamX1(op, &bitD, oend, dt, dtLog);

  if (!BIT_endOfDStream(&bitD))
    return ERROR(corruption_detected);

  return dstSize;
}

/* HUF_decompress4X1_usingDTable_internal_body():
 * Conditions :
 * @dstSize >= 6
 */
DEVICE_INLINE size_t HUF_decompress4X1_usingDTable_internal_body(
    void *dst, size_t dstSize, const void *cSrc, size_t cSrcSize,
    const HUF_DTable *DTable) {

  /* Check */
  if (cSrcSize < 10)
    return ERROR(corruption_detected); /* strict minimum : jump table + 1 byte
                                          per stream */
  if (dstSize < 6)
    return ERROR(corruption_detected); /* stream 4-split doesn't work */

  {
    const BYTE *const istart = (const BYTE *)cSrc;
    BYTE *const ostart = (BYTE *)dst;
    BYTE *const oend = ostart + dstSize;
    BYTE *const olimit = oend - 3;
    const void *const dtPtr = DTable + 1;
    const HUF_DEltX1 *const dt = (const HUF_DEltX1 *)dtPtr;

    /* Init */
    BIT_DStream_t bitD1;
    BIT_DStream_t bitD2;
    BIT_DStream_t bitD3;
    BIT_DStream_t bitD4;
    size_t const length1 = MEM_readLE16(istart);
    size_t const length2 = MEM_readLE16(istart + 2);
    size_t const length3 = MEM_readLE16(istart + 4);
    size_t const length4 = cSrcSize - (length1 + length2 + length3 + 6);
    const BYTE *const istart1 = istart + 6; /* jumpTable */
    const BYTE *const istart2 = istart1 + length1;
    const BYTE *const istart3 = istart2 + length2;
    const BYTE *const istart4 = istart3 + length3;
    const size_t segmentSize = (dstSize + 3) / 4;
    BYTE *const opStart2 = ostart + segmentSize;
    BYTE *const opStart3 = opStart2 + segmentSize;
    BYTE *const opStart4 = opStart3 + segmentSize;
    BYTE *op1 = ostart;
    BYTE *op2 = opStart2;
    BYTE *op3 = opStart3;
    BYTE *op4 = opStart4;
    DTableDesc const dtd = HUF_getDTableDesc(DTable);
    U32 const dtLog = dtd.tableLog;
    U32 endSignal = 1;

    if (length4 > cSrcSize)
      return ERROR(corruption_detected); /* overflow */
    if (opStart4 > oend)
      return ERROR(corruption_detected); /* overflow */
    assert(dstSize >= 6);                /* validated above */
    CHECK_F(BIT_initDStream(&bitD1, istart1, length1));
    CHECK_F(BIT_initDStream(&bitD2, istart2, length2));
    CHECK_F(BIT_initDStream(&bitD3, istart3, length3));
    CHECK_F(BIT_initDStream(&bitD4, istart4, length4));

    /* up to 16 symbols per loop (4 symbols per stream) in 64-bit mode */
    if ((size_t)(oend - op4) >= sizeof(size_t)) {
      for (; (endSignal) & (op4 < olimit);) {
        HUF_DECODE_SYMBOLX1_2(op1, &bitD1);
        HUF_DECODE_SYMBOLX1_2(op2, &bitD2);
        HUF_DECODE_SYMBOLX1_2(op3, &bitD3);
        HUF_DECODE_SYMBOLX1_2(op4, &bitD4);
        HUF_DECODE_SYMBOLX1_1(op1, &bitD1);
        HUF_DECODE_SYMBOLX1_1(op2, &bitD2);
        HUF_DECODE_SYMBOLX1_1(op3, &bitD3);
        HUF_DECODE_SYMBOLX1_1(op4, &bitD4);
        HUF_DECODE_SYMBOLX1_2(op1, &bitD1);
        HUF_DECODE_SYMBOLX1_2(op2, &bitD2);
        HUF_DECODE_SYMBOLX1_2(op3, &bitD3);
        HUF_DECODE_SYMBOLX1_2(op4, &bitD4);
        HUF_DECODE_SYMBOLX1_0(op1, &bitD1);
        HUF_DECODE_SYMBOLX1_0(op2, &bitD2);
        HUF_DECODE_SYMBOLX1_0(op3, &bitD3);
        HUF_DECODE_SYMBOLX1_0(op4, &bitD4);
        endSignal &= BIT_reloadDStreamFast(&bitD1) == BIT_DStream_unfinished;
        endSignal &= BIT_reloadDStreamFast(&bitD2) == BIT_DStream_unfinished;
        endSignal &= BIT_reloadDStreamFast(&bitD3) == BIT_DStream_unfinished;
        endSignal &= BIT_reloadDStreamFast(&bitD4) == BIT_DStream_unfinished;
      }
    }

    /* check corruption */
    /* note : should not be necessary : op# advance in lock step, and we control
     * op4. but curiously, binary generated by gcc 7.2 & 7.3 with -mbmi2 runs
     * faster when >=1 test is present */
    if (op1 > opStart2)
      return ERROR(corruption_detected);
    if (op2 > opStart3)
      return ERROR(corruption_detected);
    if (op3 > opStart4)
      return ERROR(corruption_detected);
    /* note : op4 supposed already verified within main loop */

    /* finish bitStreams one by one */
    HUF_decodeStreamX1(op1, &bitD1, opStart2, dt, dtLog);
    HUF_decodeStreamX1(op2, &bitD2, opStart3, dt, dtLog);
    HUF_decodeStreamX1(op3, &bitD3, opStart4, dt, dtLog);
    HUF_decodeStreamX1(op4, &bitD4, oend, dt, dtLog);

    /* check */
    {
      U32 const endCheck = BIT_endOfDStream(&bitD1) & BIT_endOfDStream(&bitD2) &
                           BIT_endOfDStream(&bitD3) & BIT_endOfDStream(&bitD4);
      if (!endCheck)
        return ERROR(corruption_detected);
    }

    /* decoded size */
    return dstSize;
  }
}

DEVICE_INLINE size_t HUF_decompress4X1_usingDTable_internal_bmi2(
    void *dst, size_t dstSize, void const *cSrc, size_t cSrcSize,
    HUF_DTable const *DTable) {
  return HUF_decompress4X1_usingDTable_internal_body(dst, dstSize, cSrc,
                                                     cSrcSize, DTable);
}

DEVICE_INLINE size_t HUF_decompress4X1_usingDTable_internal_default(
    void *dst, size_t dstSize, void const *cSrc, size_t cSrcSize,
    HUF_DTable const *DTable) {
  return HUF_decompress4X1_usingDTable_internal_body(dst, dstSize, cSrc,
                                                     cSrcSize, DTable);
}

DEVICE_INLINE void HUF_decompress4X1_usingDTable_internal_fast_c_loop(
    HUF_DecompressFastArgs *args) {
  U64 bits[4];
  BYTE const *ip[4];
  BYTE *op[4];
  U16 const *const dtable = (U16 const *)args->dt;
  BYTE *const oend = args->oend;
  BYTE const *const ilowest = args->ilowest;

  /* Copy the arguments to local variables */
  ZSTD_memcpy(&bits, &args->bits, sizeof(bits));
  ZSTD_memcpy((void *)(&ip), &args->ip, sizeof(ip));
  ZSTD_memcpy(&op, &args->op, sizeof(op));

  assert(MEM_isLittleEndian());
  assert(!MEM_32bits());

  for (;;) {
    BYTE *olimit;
    int stream;

    /* Assert loop preconditions */
#ifndef NDEBUG
    for (stream = 0; stream < 4; ++stream) {
      assert(op[stream] <= (stream == 3 ? oend : op[stream + 1]));
      assert(ip[stream] >= ilowest);
    }
#endif
    /* Compute olimit */
    {
      /* Each iteration produces 5 output symbols per stream */
      size_t const oiters = (size_t)(oend - op[3]) / 5;
      /* Each iteration consumes up to 11 bits * 5 = 55 bits < 7 bytes
       * per stream.
       */
      size_t const iiters = (size_t)(ip[0] - ilowest) / 7;
      /* We can safely run iters iterations before running bounds checks */
      size_t const iters = MIN(oiters, iiters);
      size_t const symbols = iters * 5;

      /* We can simply check that op[3] < olimit, instead of checking all
       * of our bounds, since we can't hit the other bounds until we've run
       * iters iterations, which only happens when op[3] == olimit.
       */
      olimit = op[3] + symbols;

      /* Exit fast decoding loop once we reach the end. */
      if (op[3] == olimit)
        break;

      /* Exit the decoding loop if any input pointer has crossed the
       * previous one. This indicates corruption, and a precondition
       * to our loop is that ip[i] >= ip[0].
       */
      for (stream = 1; stream < 4; ++stream) {
        if (ip[stream] < ip[stream - 1])
          goto _out;
      }
    }

#ifndef NDEBUG
    for (stream = 1; stream < 4; ++stream) {
      assert(ip[stream] >= ip[stream - 1]);
    }
#endif

#define HUF_4X1_DECODE_SYMBOL(_stream, _symbol)                                \
  do {                                                                         \
    U64 const index = bits[(_stream)] >> 53;                                   \
    U16 const entry = dtable[index];                                           \
    bits[(_stream)] <<= entry & 0x3F;                                          \
    op[(_stream)][(_symbol)] = (BYTE)(entry >> 8);                             \
  } while (0)

#define HUF_5X1_RELOAD_STREAM(_stream)                                         \
  do {                                                                         \
    U64 const ctz = ZSTD_countTrailingZeros64(bits[(_stream)]);                \
    U64 const nbBits = ctz & 7;                                                \
    U64 const nbBytes = ctz >> 3;                                              \
    op[(_stream)] += 5;                                                        \
    ip[(_stream)] -= nbBytes;                                                  \
    bits[(_stream)] = MEM_read64(ip[(_stream)]) | 1;                           \
    bits[(_stream)] <<= nbBits;                                                \
  } while (0)

    /* Manually unroll the loop because compilers don't consistently
     * unroll the inner loops, which destroys performance.
     */
    do {
      /* Decode 5 symbols in each of the 4 streams */
      HUF_4X_FOR_EACH_STREAM_WITH_VAR(HUF_4X1_DECODE_SYMBOL, 0);
      HUF_4X_FOR_EACH_STREAM_WITH_VAR(HUF_4X1_DECODE_SYMBOL, 1);
      HUF_4X_FOR_EACH_STREAM_WITH_VAR(HUF_4X1_DECODE_SYMBOL, 2);
      HUF_4X_FOR_EACH_STREAM_WITH_VAR(HUF_4X1_DECODE_SYMBOL, 3);
      HUF_4X_FOR_EACH_STREAM_WITH_VAR(HUF_4X1_DECODE_SYMBOL, 4);

      /* Reload each of the 4 the bitstreams */
      HUF_4X_FOR_EACH_STREAM(HUF_5X1_RELOAD_STREAM);
    } while (op[3] < olimit);

#undef HUF_4X1_DECODE_SYMBOL
#undef HUF_5X1_RELOAD_STREAM
  }

_out:

  /* Save the final values of each of the state variables back to args. */
  ZSTD_memcpy(&args->bits, &bits, sizeof(bits));
  ZSTD_memcpy((void *)(&args->ip), &ip, sizeof(ip));
  ZSTD_memcpy(&args->op, &op, sizeof(op));
}

/**
 * @returns @p dstSize on success (>= 6)
 *          0 if the fallback implementation should be used
 *          An error if an error occurred
 */
DEVICE_INLINE size_t HUF_decompress4X1_usingDTable_internal_fast(
    void *dst, size_t dstSize, const void *cSrc, size_t cSrcSize,
    const HUF_DTable *DTable, HUF_DecompressFastLoopFn loopFn) {
  void const *dt = DTable + 1;
  BYTE const *const ilowest = (BYTE const *)cSrc;
  BYTE *const oend = (BYTE *)ZSTD_maybeNullPtrAdd(dst, (ptrdiff_t)dstSize);
  HUF_DecompressFastArgs args;
  {
    size_t const ret = HUF_DecompressFastArgs_init(&args, dst, dstSize, cSrc,
                                                   cSrcSize, DTable);
    FORWARD_IF_ERROR(ret, "Failed to init fast loop args");
    if (ret == 0)
      return 0;
  }

  assert(args.ip[0] >= args.ilowest);
  loopFn(&args); // TODO(HIP/AMD): make template parm

  /* Our loop guarantees that ip[] >= ilowest and that we haven't
   * overwritten any op[].
   */
  assert(args.ip[0] >= ilowest);
  assert(args.ip[0] >= ilowest);
  assert(args.ip[1] >= ilowest);
  assert(args.ip[2] >= ilowest);
  assert(args.ip[3] >= ilowest);
  assert(args.op[3] <= oend);

  assert(ilowest == args.ilowest);
  assert(ilowest + 6 == args.iend[0]);
  (void)ilowest;

  /* finish bit streams one by one. */
  {
    size_t const segmentSize = (dstSize + 3) / 4;
    BYTE *segmentEnd = (BYTE *)dst;
    int i;
    for (i = 0; i < 4; ++i) {
      BIT_DStream_t bit;
      if (segmentSize <= (size_t)(oend - segmentEnd))
        segmentEnd += segmentSize;
      else
        segmentEnd = oend;
      FORWARD_IF_ERROR(HUF_initRemainingDStream(&bit, &args, i, segmentEnd),
                       "corruption");
      /* Decompress and validate that we've produced exactly the expected
       * length. */
      args.op[i] +=
          HUF_decodeStreamX1(args.op[i], &bit, segmentEnd,
                             (HUF_DEltX1 const *)dt, HUF_DECODER_FAST_TABLELOG);
      if (args.op[i] != segmentEnd)
        return ERROR(corruption_detected);
    }
  }

  /* decoded size */
  assert(dstSize != 0);
  return dstSize;
}

DEVICE_INLINE size_t HUF_decompress1X1_usingDTable_internal_default(
    void *dst, size_t dstSize, const void *cSrc, size_t cSrcSize,
    const HUF_DTable *DTable) {
  return HUF_decompress1X1_usingDTable_internal_body(dst, dstSize, cSrc,
                                                     cSrcSize, DTable);
}
DEVICE_INLINE size_t HUF_decompress1X1_usingDTable_internal_bmi2(
    void *dst, size_t dstSize, const void *cSrc, size_t cSrcSize,
    const HUF_DTable *DTable) {
  return HUF_decompress1X1_usingDTable_internal_body(dst, dstSize, cSrc,
                                                     cSrcSize, DTable);
}
DEVICE_INLINE size_t HUF_decompress1X1_usingDTable_internal(
    void *dst, size_t dstSize, void const *cSrc, size_t cSrcSize,
    HUF_DTable const *DTable, int flags) {
  if (flags & HUF_flags_bmi2) {
    return HUF_decompress1X1_usingDTable_internal_bmi2(dst, dstSize, cSrc,
                                                       cSrcSize, DTable);
  }
  return HUF_decompress1X1_usingDTable_internal_default(dst, dstSize, cSrc,
                                                        cSrcSize, DTable);
}

DEVICE_INLINE size_t HUF_decompress4X1_usingDTable_internal(
    void *dst, size_t dstSize, void const *cSrc, size_t cSrcSize,
    HUF_DTable const *DTable, int flags) {
  HUF_DecompressUsingDTableFn fallbackFn =
      HUF_decompress4X1_usingDTable_internal_default; // TODO(HIP/AMD): replace
                                                      // by template fn
  HUF_DecompressFastLoopFn loopFn =
      HUF_decompress4X1_usingDTable_internal_fast_c_loop; // TODO(HIP/AMD):
                                                          // replace by template
                                                          // fn

#if DYNAMIC_BMI2
  if (flags & HUF_flags_bmi2) {
    fallbackFn = HUF_decompress4X1_usingDTable_internal_bmi2;
#if ZSTD_ENABLE_ASM_X86_64_BMI2
    if (!(flags & HUF_flags_disableAsm)) {
      loopFn = HUF_decompress4X1_usingDTable_internal_fast_asm_loop;
    }
#endif
  } else {
    return fallbackFn(dst, dstSize, cSrc, cSrcSize, DTable);
  }
#endif

#if ZSTD_ENABLE_ASM_X86_64_BMI2 && defined(__BMI2__)
  if (!(flags & HUF_flags_disableAsm)) {
    loopFn = HUF_decompress4X1_usingDTable_internal_fast_asm_loop;
  }
#endif

  if (HUF_ENABLE_FAST_DECODE && !(flags & HUF_flags_disableFast)) {
    size_t const ret = HUF_decompress4X1_usingDTable_internal_fast(
        dst, dstSize, cSrc, cSrcSize, DTable, loopFn);
    if (ret != 0)
      return ret;
  }
  return fallbackFn(dst, dstSize, cSrc, cSrcSize,
                    DTable); // TODO(HIP/AMD): replace by template fn
}

DEVICE_INLINE
size_t HUF_decompress4X1_DCtx_wksp(HUF_DTable *dctx, void *dst, size_t dstSize,
                                   const void *cSrc, size_t cSrcSize,
                                   void *workSpace, size_t wkspSize,
                                   int flags) {
  const BYTE *ip = (const BYTE *)cSrc;

  size_t const hSize =
      HUF_readDTableX1_wksp(dctx, cSrc, cSrcSize, workSpace, wkspSize, flags);
  if (ERR_isError(hSize))
    return hSize;
  if (hSize >= cSrcSize)
    return ((size_t)-ZSTD_error_srcSize_wrong);
  ip += hSize;
  cSrcSize -= hSize;

  return HUF_decompress4X1_usingDTable_internal(dst, dstSize, ip, cSrcSize,
                                                dctx, flags);
}

typedef struct {
  U16 sequence;
  BYTE nbBits;
  BYTE length;
} HUF_DEltX2;

typedef struct {
  BYTE symbol;
} sortedSymbol_t;

typedef U32 rankValCol_t[12 + 1];

DEVICE_INLINE U32 HUF_buildDEltX2U32(U32 symbol, U32 nbBits, U32 baseSeq,
                                     int level) {
  U32 seq;
  (void)sizeof(char[(__builtin_offsetof(HUF_DEltX2, sequence) == 0) ? 1 : -1]);
  (void)sizeof(char[(__builtin_offsetof(HUF_DEltX2, nbBits) == 2) ? 1 : -1]);
  (void)sizeof(char[(__builtin_offsetof(HUF_DEltX2, length) == 3) ? 1 : -1]);
  (void)sizeof(char[(sizeof(HUF_DEltX2) == sizeof(U32)) ? 1 : -1]);
  if (MEM_isLittleEndian()) {
    seq = level == 1 ? symbol : (baseSeq + (symbol << 8));
    return seq + (nbBits << 16) + ((U32)level << 24);
  } else {
    seq = level == 1 ? (symbol << 8) : ((baseSeq << 8) + symbol);
    return (seq << 16) + (nbBits << 8) + (U32)level;
  }
}

DEVICE_INLINE HUF_DEltX2 HUF_buildDEltX2(U32 symbol, U32 nbBits, U32 baseSeq,
                                         int level) {
  HUF_DEltX2 DElt;
  U32 const val = HUF_buildDEltX2U32(symbol, nbBits, baseSeq, level);
  (void)sizeof(char[(sizeof(DElt) == sizeof(val)) ? 1 : -1]);
  __builtin_memcpy((&DElt), (&val), (sizeof(val)));
  return DElt;
}

DEVICE_INLINE U64 HUF_buildDEltX2U64(U32 symbol, U32 nbBits, U16 baseSeq,
                                     int level) {
  U32 DElt = HUF_buildDEltX2U32(symbol, nbBits, baseSeq, level);
  return (U64)DElt + ((U64)DElt << 32);
}

DEVICE_INLINE void HUF_fillDTableX2ForWeight(HUF_DEltX2 *DTableRank,
                                             sortedSymbol_t const *begin,
                                             sortedSymbol_t const *end,
                                             U32 nbBits, U32 tableLog,
                                             U16 baseSeq, int const level) {
  U32 const length = 1U << ((tableLog - nbBits) & 0x1F);
  const sortedSymbol_t *ptr;

  switch (length) {
  case 1:
    for (ptr = begin; ptr != end; ++ptr) {
      HUF_DEltX2 const DElt =
          HUF_buildDEltX2(ptr->symbol, nbBits, baseSeq, level);
      *DTableRank++ = DElt;
    }
    break;
  case 2:
    for (ptr = begin; ptr != end; ++ptr) {
      HUF_DEltX2 const DElt =
          HUF_buildDEltX2(ptr->symbol, nbBits, baseSeq, level);
      DTableRank[0] = DElt;
      DTableRank[1] = DElt;
      DTableRank += 2;
    }
    break;
  case 4:
    for (ptr = begin; ptr != end; ++ptr) {
      U64 const DEltX2 =
          HUF_buildDEltX2U64(ptr->symbol, nbBits, baseSeq, level);
      __builtin_memcpy((DTableRank + 0), (&DEltX2), (sizeof(DEltX2)));
      __builtin_memcpy((DTableRank + 2), (&DEltX2), (sizeof(DEltX2)));
      DTableRank += 4;
    }
    break;
  case 8:
    for (ptr = begin; ptr != end; ++ptr) {
      U64 const DEltX2 =
          HUF_buildDEltX2U64(ptr->symbol, nbBits, baseSeq, level);
      __builtin_memcpy((DTableRank + 0), (&DEltX2), (sizeof(DEltX2)));
      __builtin_memcpy((DTableRank + 2), (&DEltX2), (sizeof(DEltX2)));
      __builtin_memcpy((DTableRank + 4), (&DEltX2), (sizeof(DEltX2)));
      __builtin_memcpy((DTableRank + 6), (&DEltX2), (sizeof(DEltX2)));
      DTableRank += 8;
    }
    break;
  default:
    for (ptr = begin; ptr != end; ++ptr) {
      U64 const DEltX2 =
          HUF_buildDEltX2U64(ptr->symbol, nbBits, baseSeq, level);
      HUF_DEltX2 *const DTableRankEnd = DTableRank + length;
      for (; DTableRank != DTableRankEnd; DTableRank += 8) {
        __builtin_memcpy((DTableRank + 0), (&DEltX2), (sizeof(DEltX2)));
        __builtin_memcpy((DTableRank + 2), (&DEltX2), (sizeof(DEltX2)));
        __builtin_memcpy((DTableRank + 4), (&DEltX2), (sizeof(DEltX2)));
        __builtin_memcpy((DTableRank + 6), (&DEltX2), (sizeof(DEltX2)));
      }
    }
    break;
  }
}

DEVICE_INLINE void
HUF_fillDTableX2Level2(HUF_DEltX2 *DTable, U32 targetLog,
                       const U32 consumedBits, const U32 *rankVal,
                       const int minWeight, const int maxWeight1,
                       const sortedSymbol_t *sortedSymbols,
                       U32 const *rankStart, U32 nbBitsBaseline, U16 baseSeq) {

  if (minWeight > 1) {
    U32 const length = 1U << ((targetLog - consumedBits) & 0x1F);
    U64 const DEltX2 = HUF_buildDEltX2U64(baseSeq, consumedBits, 0, 1);
    int const skipSize = rankVal[minWeight];

    switch (length) {
    case 2:

      __builtin_memcpy((DTable), (&DEltX2), (sizeof(DEltX2)));
      break;
    case 4:

      __builtin_memcpy((DTable + 0), (&DEltX2), (sizeof(DEltX2)));
      __builtin_memcpy((DTable + 2), (&DEltX2), (sizeof(DEltX2)));
      break;
    default: {
      int i;
      for (i = 0; i < skipSize; i += 8) {
        __builtin_memcpy((DTable + i + 0), (&DEltX2), (sizeof(DEltX2)));
        __builtin_memcpy((DTable + i + 2), (&DEltX2), (sizeof(DEltX2)));
        __builtin_memcpy((DTable + i + 4), (&DEltX2), (sizeof(DEltX2)));
        __builtin_memcpy((DTable + i + 6), (&DEltX2), (sizeof(DEltX2)));
      }
    }
    }
  }

  {
    int w;
    for (w = minWeight; w < maxWeight1; ++w) {
      int const begin = rankStart[w];
      int const end = rankStart[w + 1];
      U32 const nbBits = nbBitsBaseline - w;
      U32 const totalBits = nbBits + consumedBits;
      HUF_fillDTableX2ForWeight(DTable + rankVal[w], sortedSymbols + begin,
                                sortedSymbols + end, totalBits, targetLog,
                                baseSeq, 2);
    }
  }
}

DEVICE_INLINE void HUF_fillDTableX2(HUF_DEltX2 *DTable, const U32 targetLog,
                                    const sortedSymbol_t *sortedList,
                                    const U32 *rankStart,
                                    rankValCol_t *rankValOrigin,
                                    const U32 maxWeight,
                                    const U32 nbBitsBaseline) {
  U32 *const rankVal = rankValOrigin[0];
  const int scaleLog = nbBitsBaseline - targetLog;
  const U32 minBits = nbBitsBaseline - maxWeight;
  int w;
  int const wEnd = (int)maxWeight + 1;

  for (w = 1; w < wEnd; ++w) {
    int const begin = (int)rankStart[w];
    int const end = (int)rankStart[w + 1];
    U32 const nbBits = nbBitsBaseline - w;

    if (targetLog - nbBits >= minBits) {

      int start = rankVal[w];
      U32 const length = 1U << ((targetLog - nbBits) & 0x1F);
      int minWeight = nbBits + scaleLog;
      int s;
      if (minWeight < 1)
        minWeight = 1;

      for (s = begin; s != end; ++s) {
        HUF_fillDTableX2Level2(
            DTable + start, targetLog, nbBits, rankValOrigin[nbBits], minWeight,
            wEnd, sortedList, rankStart, nbBitsBaseline, sortedList[s].symbol);
        start += length;
      }
    } else {

      HUF_fillDTableX2ForWeight(DTable + rankVal[w], sortedList + begin,
                                sortedList + end, nbBits, targetLog, 0, 1);
    }
  }
}

typedef struct {
  rankValCol_t rankVal[12];
  U32 rankStats[12 + 1];
  U32 rankStart0[12 + 3];
  sortedSymbol_t sortedSymbol[255 + 1];
  BYTE weightList[255 + 1];
  U32 calleeWksp[((1 + (1 << (6))) + 1 +
                  (((sizeof(short) * (12 - 1 + 1) + (1ULL << 6) + 8) +
                    sizeof(unsigned) - 1) /
                   sizeof(unsigned)) +
                  (255 + 1) / 2 + 1)];
} HUF_ReadDTableX2_Workspace;

DEVICE_INLINE size_t HUF_readDTableX2_wksp(HUF_DTable *DTable, const void *src,
                                           size_t srcSize, void *workSpace,
                                           size_t wkspSize, int flags) {
  U32 tableLog, maxW, nbSymbols;
  DTableDesc dtd = HUF_getDTableDesc(DTable);
  U32 maxTableLog = dtd.maxTableLog;
  size_t iSize;
  void *dtPtr = DTable + 1;
  HUF_DEltX2 *const dt = (HUF_DEltX2 *)dtPtr;
  U32 *rankStart;

  HUF_ReadDTableX2_Workspace *const wksp =
      (HUF_ReadDTableX2_Workspace *)workSpace;

  if (sizeof(*wksp) > wkspSize)
    return ((size_t)-ZSTD_error_GENERIC);

  rankStart = wksp->rankStart0 + 1;
  __builtin_memset((wksp->rankStats), (0), (sizeof(wksp->rankStats)));
  __builtin_memset((wksp->rankStart0), (0), (sizeof(wksp->rankStart0)));

  (void)sizeof(char[(sizeof(HUF_DEltX2) == sizeof(HUF_DTable)) ? 1 : -1]);
  if (maxTableLog > 12)
    return ((size_t)-ZSTD_error_tableLog_tooLarge);

  iSize = HUF_readStats_wksp(wksp->weightList, 255 + 1, wksp->rankStats,
                             &nbSymbols, &tableLog, src, srcSize,
                             wksp->calleeWksp, sizeof(wksp->calleeWksp), flags);
  if (ERR_isError(iSize))
    return iSize;

  if (tableLog > maxTableLog)
    return ((size_t)-ZSTD_error_tableLog_tooLarge);
  if (tableLog <= 11 && maxTableLog > 11)
    maxTableLog = 11;

  for (maxW = tableLog; wksp->rankStats[maxW] == 0; maxW--) {
  }

  {
    U32 w, nextRankStart = 0;
    for (w = 1; w < maxW + 1; w++) {
      U32 curr = nextRankStart;
      nextRankStart += wksp->rankStats[w];
      rankStart[w] = curr;
    }
    rankStart[0] = nextRankStart;
    rankStart[maxW + 1] = nextRankStart;
  }

  {
    U32 s;
    for (s = 0; s < nbSymbols; s++) {
      U32 const w = wksp->weightList[s];
      U32 const r = rankStart[w]++;
      wksp->sortedSymbol[r].symbol = (BYTE)s;
    }
    rankStart[0] = 0;
  }

  {
    U32 *const rankVal0 = wksp->rankVal[0];
    {
      int const rescale = (maxTableLog - tableLog) - 1;
      U32 nextRankVal = 0;
      U32 w;
      for (w = 1; w < maxW + 1; w++) {
        U32 curr = nextRankVal;
        nextRankVal += wksp->rankStats[w] << (w + rescale);
        rankVal0[w] = curr;
      }
    }
    {
      U32 const minBits = tableLog + 1 - maxW;
      U32 consumed;
      for (consumed = minBits; consumed < maxTableLog - minBits + 1;
           consumed++) {
        U32 *const rankValPtr = wksp->rankVal[consumed];
        U32 w;
        for (w = 1; w < maxW + 1; w++) {
          rankValPtr[w] = rankVal0[w] >> consumed;
        }
      }
    }
  }

  HUF_fillDTableX2(dt, maxTableLog, wksp->sortedSymbol, wksp->rankStart0,
                   wksp->rankVal, maxW, tableLog + 1);

  dtd.tableLog = (BYTE)maxTableLog;
  dtd.tableType = 1;
  __builtin_memcpy((DTable), (&dtd), (sizeof(dtd)));
  return iSize;
}

DEVICE_INLINE U32 HUF_decodeSymbolX2(void *op, BIT_DStream_t *DStream,
                                     const HUF_DEltX2 *dt, const U32 dtLog) {
  size_t const val = BIT_lookBitsFast(DStream, dtLog);
  __builtin_memcpy((op), (&dt[val].sequence), (2));
  BIT_skipBits(DStream, dt[val].nbBits);
  return dt[val].length;
}

DEVICE_INLINE U32 HUF_decodeLastSymbolX2(void *op, BIT_DStream_t *DStream,
                                         const HUF_DEltX2 *dt,
                                         const U32 dtLog) {
  size_t const val = BIT_lookBitsFast(DStream, dtLog);
  __builtin_memcpy((op), (&dt[val].sequence), (1));
  if (dt[val].length == 1) {
    BIT_skipBits(DStream, dt[val].nbBits);
  } else {
    if (DStream->bitsConsumed < (sizeof(DStream->bitContainer) * 8)) {
      BIT_skipBits(DStream, dt[val].nbBits);
      if (DStream->bitsConsumed > (sizeof(DStream->bitContainer) * 8))

        DStream->bitsConsumed = (sizeof(DStream->bitContainer) * 8);
    }
  }
  return 1;
}

DEVICE_INLINE size_t HUF_decodeStreamX2(BYTE *p, BIT_DStream_t *bitDPtr,
                                        BYTE *const pEnd,
                                        const HUF_DEltX2 *const dt,
                                        const U32 dtLog) {
  BYTE *const pStart = p;

  if ((size_t)(pEnd - p) >= sizeof(bitDPtr->bitContainer)) {
    if (dtLog <= 11 && MEM_64bits()) {

      while ((BIT_reloadDStream(bitDPtr) == BIT_DStream_unfinished) &
             (p < pEnd - 9)) {
        do {
          p += HUF_decodeSymbolX2(p, bitDPtr, dt, dtLog);
        } while (0);
        do {
          p += HUF_decodeSymbolX2(p, bitDPtr, dt, dtLog);
        } while (0);
        do {
          p += HUF_decodeSymbolX2(p, bitDPtr, dt, dtLog);
        } while (0);
        do {
          p += HUF_decodeSymbolX2(p, bitDPtr, dt, dtLog);
        } while (0);
        do {
          p += HUF_decodeSymbolX2(p, bitDPtr, dt, dtLog);
        } while (0);
      }
    } else {

      while ((BIT_reloadDStream(bitDPtr) == BIT_DStream_unfinished) &
             (p < pEnd - (sizeof(bitDPtr->bitContainer) - 1))) {
        do {
          if (MEM_64bits())
            p += HUF_decodeSymbolX2(p, bitDPtr, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits() || (12 <= 12))
            p += HUF_decodeSymbolX2(p, bitDPtr, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits())
            p += HUF_decodeSymbolX2(p, bitDPtr, dt, dtLog);
        } while (0);
        do {
          p += HUF_decodeSymbolX2(p, bitDPtr, dt, dtLog);
        } while (0);
      }
    }
  } else {
    BIT_reloadDStream(bitDPtr);
  }

  if ((size_t)(pEnd - p) >= 2) {
    while ((BIT_reloadDStream(bitDPtr) == BIT_DStream_unfinished) &
           (p <= pEnd - 2))
      do {
        p += HUF_decodeSymbolX2(p, bitDPtr, dt, dtLog);
      } while (0);

    while (p <= pEnd - 2)
      do {
        p += HUF_decodeSymbolX2(p, bitDPtr, dt, dtLog);
      } while (0);
  }

  if (p < pEnd)
    p += HUF_decodeLastSymbolX2(p, bitDPtr, dt, dtLog);

  return p - pStart;
}

DEVICE_INLINE size_t HUF_decompress1X2_usingDTable_internal_body(
    void *dst, size_t dstSize, const void *cSrc, size_t cSrcSize,
    const HUF_DTable *DTable) {
  BIT_DStream_t bitD;

  do {
    size_t const _var_err__ = BIT_initDStream(&bitD, cSrc, cSrcSize);
    do {
      if (ERR_isError(_var_err__))
        return _var_err__;
    } while (0);
  } while (0);

  {
    BYTE *const ostart = (BYTE *)dst;
    BYTE *const oend = (BYTE *)ZSTD_maybeNullPtrAdd(ostart, (ptrdiff_t)dstSize);
    const void *const dtPtr = DTable + 1;
    const HUF_DEltX2 *const dt = (const HUF_DEltX2 *)dtPtr;
    DTableDesc const dtd = HUF_getDTableDesc(DTable);
    HUF_decodeStreamX2(ostart, &bitD, oend, dt, dtd.tableLog);
  }

  if (!BIT_endOfDStream(&bitD))
    return ((size_t)-ZSTD_error_corruption_detected);

  return dstSize;
}

DEVICE_INLINE size_t HUF_decompress4X2_usingDTable_internal_body(
    void *dst, size_t dstSize, const void *cSrc, size_t cSrcSize,
    const HUF_DTable *DTable) {
  if (cSrcSize < 10)
    return ((size_t)-ZSTD_error_corruption_detected);
  if (dstSize < 6)
    return ((size_t)-ZSTD_error_corruption_detected);

  {
    const BYTE *const istart = (const BYTE *)cSrc;
    BYTE *const ostart = (BYTE *)dst;
    BYTE *const oend = ostart + dstSize;
    BYTE *const olimit = oend - (sizeof(size_t) - 1);
    const void *const dtPtr = DTable + 1;
    const HUF_DEltX2 *const dt = (const HUF_DEltX2 *)dtPtr;

    BIT_DStream_t bitD1;
    BIT_DStream_t bitD2;
    BIT_DStream_t bitD3;
    BIT_DStream_t bitD4;
    size_t const length1 = MEM_readLE16(istart);
    size_t const length2 = MEM_readLE16(istart + 2);
    size_t const length3 = MEM_readLE16(istart + 4);
    size_t const length4 = cSrcSize - (length1 + length2 + length3 + 6);
    const BYTE *const istart1 = istart + 6;
    const BYTE *const istart2 = istart1 + length1;
    const BYTE *const istart3 = istart2 + length2;
    const BYTE *const istart4 = istart3 + length3;
    size_t const segmentSize = (dstSize + 3) / 4;
    BYTE *const opStart2 = ostart + segmentSize;
    BYTE *const opStart3 = opStart2 + segmentSize;
    BYTE *const opStart4 = opStart3 + segmentSize;
    BYTE *op1 = ostart;
    BYTE *op2 = opStart2;
    BYTE *op3 = opStart3;
    BYTE *op4 = opStart4;
    U32 endSignal = 1;
    DTableDesc const dtd = HUF_getDTableDesc(DTable);
    U32 const dtLog = dtd.tableLog;

    if (length4 > cSrcSize)
      return ((size_t)-ZSTD_error_corruption_detected);
    if (opStart4 > oend)
      return ((size_t)-ZSTD_error_corruption_detected);

    do {
      size_t const _var_err__ = BIT_initDStream(&bitD1, istart1, length1);
      do {
        if (ERR_isError(_var_err__))
          return _var_err__;
      } while (0);
    } while (0);
    do {
      size_t const _var_err__ = BIT_initDStream(&bitD2, istart2, length2);
      do {
        if (ERR_isError(_var_err__))
          return _var_err__;
      } while (0);
    } while (0);
    do {
      size_t const _var_err__ = BIT_initDStream(&bitD3, istart3, length3);
      do {
        if (ERR_isError(_var_err__))
          return _var_err__;
      } while (0);
    } while (0);
    do {
      size_t const _var_err__ = BIT_initDStream(&bitD4, istart4, length4);
      do {
        if (ERR_isError(_var_err__))
          return _var_err__;
      } while (0);
    } while (0);

    if ((size_t)(oend - op4) >= sizeof(size_t)) {
      for (; (endSignal) & (op4 < olimit);) {
        do {
          if (MEM_64bits())
            op1 += HUF_decodeSymbolX2(op1, &bitD1, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits())
            op2 += HUF_decodeSymbolX2(op2, &bitD2, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits())
            op3 += HUF_decodeSymbolX2(op3, &bitD3, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits())
            op4 += HUF_decodeSymbolX2(op4, &bitD4, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits() || (12 <= 12))
            op1 += HUF_decodeSymbolX2(op1, &bitD1, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits() || (12 <= 12))
            op2 += HUF_decodeSymbolX2(op2, &bitD2, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits() || (12 <= 12))
            op3 += HUF_decodeSymbolX2(op3, &bitD3, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits() || (12 <= 12))
            op4 += HUF_decodeSymbolX2(op4, &bitD4, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits())
            op1 += HUF_decodeSymbolX2(op1, &bitD1, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits())
            op2 += HUF_decodeSymbolX2(op2, &bitD2, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits())
            op3 += HUF_decodeSymbolX2(op3, &bitD3, dt, dtLog);
        } while (0);
        do {
          if (MEM_64bits())
            op4 += HUF_decodeSymbolX2(op4, &bitD4, dt, dtLog);
        } while (0);
        do {
          op1 += HUF_decodeSymbolX2(op1, &bitD1, dt, dtLog);
        } while (0);
        do {
          op2 += HUF_decodeSymbolX2(op2, &bitD2, dt, dtLog);
        } while (0);
        do {
          op3 += HUF_decodeSymbolX2(op3, &bitD3, dt, dtLog);
        } while (0);
        do {
          op4 += HUF_decodeSymbolX2(op4, &bitD4, dt, dtLog);
        } while (0);
        endSignal = (U32)(__builtin_expect(
            ((U32)(BIT_reloadDStreamFast(&bitD1) == BIT_DStream_unfinished) &
             (BIT_reloadDStreamFast(&bitD2) == BIT_DStream_unfinished) &
             (BIT_reloadDStreamFast(&bitD3) == BIT_DStream_unfinished) &
             (BIT_reloadDStreamFast(&bitD4) == BIT_DStream_unfinished)),
            1))

            ;
      }
    }

    if (op1 > opStart2)
      return ((size_t)-ZSTD_error_corruption_detected);
    if (op2 > opStart3)
      return ((size_t)-ZSTD_error_corruption_detected);
    if (op3 > opStart4)
      return ((size_t)-ZSTD_error_corruption_detected);

    HUF_decodeStreamX2(op1, &bitD1, opStart2, dt, dtLog);
    HUF_decodeStreamX2(op2, &bitD2, opStart3, dt, dtLog);
    HUF_decodeStreamX2(op3, &bitD3, opStart4, dt, dtLog);
    HUF_decodeStreamX2(op4, &bitD4, oend, dt, dtLog);

    {
      U32 const endCheck = BIT_endOfDStream(&bitD1) & BIT_endOfDStream(&bitD2) &
                           BIT_endOfDStream(&bitD3) & BIT_endOfDStream(&bitD4);
      if (!endCheck)
        return ((size_t)-ZSTD_error_corruption_detected);
    }

    return dstSize;
  }
}

DEVICE_INLINE size_t HUF_decompress4X2_usingDTable_internal_bmi2(
    void *dst, size_t dstSize, void const *cSrc, size_t cSrcSize,
    HUF_DTable const *DTable) {
  return HUF_decompress4X2_usingDTable_internal_body(dst, dstSize, cSrc,
                                                     cSrcSize, DTable);
}

DEVICE_INLINE size_t HUF_decompress4X2_usingDTable_internal_default(
    void *dst, size_t dstSize, void const *cSrc, size_t cSrcSize,
    HUF_DTable const *DTable) {
  return HUF_decompress4X2_usingDTable_internal_body(dst, dstSize, cSrc,
                                                     cSrcSize, DTable);
}

DEVICE_INLINE void HUF_decompress4X2_usingDTable_internal_fast_c_loop(
    HUF_DecompressFastArgs *args) {
  U64 bits[4];
  BYTE const *ip[4];
  BYTE *op[4];
  BYTE *oend[4];
  HUF_DEltX2 const *const dtable = (HUF_DEltX2 const *)args->dt;
  BYTE const *const ilowest = args->ilowest;

  __builtin_memcpy((&bits), (&args->bits), (sizeof(bits)));
  __builtin_memcpy(((void *)(&ip)), (&args->ip), (sizeof(ip)));
  __builtin_memcpy((&op), (&args->op), (sizeof(op)));

  oend[0] = op[1];
  oend[1] = op[2];
  oend[2] = op[3];
  oend[3] = args->oend;

  for (;;) {
    BYTE *olimit;
    int stream;
    {
      size_t iters = (size_t)(ip[0] - ilowest) / 7;

      for (stream = 0; stream < 4; ++stream) {
        size_t const oiters = (size_t)(oend[stream] - op[stream]) / 10;
        iters = ((iters) < (oiters) ? (iters) : (oiters));
      }

      olimit = op[3] + (iters * 5);

      if (op[3] == olimit)
        break;

      for (stream = 1; stream < 4; ++stream) {
        if (ip[stream] < ip[stream - 1])
          goto _out;
      }
    }
    do {

      do {
        do {
          if (((0)) || (0) != 3) {
            U64 const index = bits[(0)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(0)], (U16)entry);
            bits[(0)] <<= (entry >> 16) & 0x3F;
            op[(0)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (1) != 3) {
            U64 const index = bits[(1)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(1)], (U16)entry);
            bits[(1)] <<= (entry >> 16) & 0x3F;
            op[(1)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (2) != 3) {
            U64 const index = bits[(2)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(2)], (U16)entry);
            bits[(2)] <<= (entry >> 16) & 0x3F;
            op[(2)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (3) != 3) {
            U64 const index = bits[(3)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(3)], (U16)entry);
            bits[(3)] <<= (entry >> 16) & 0x3F;
            op[(3)] += entry >> 24;
          }
        } while (0);
      } while (0);
      do {
        do {
          if (((0)) || (0) != 3) {
            U64 const index = bits[(0)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(0)], (U16)entry);
            bits[(0)] <<= (entry >> 16) & 0x3F;
            op[(0)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (1) != 3) {
            U64 const index = bits[(1)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(1)], (U16)entry);
            bits[(1)] <<= (entry >> 16) & 0x3F;
            op[(1)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (2) != 3) {
            U64 const index = bits[(2)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(2)], (U16)entry);
            bits[(2)] <<= (entry >> 16) & 0x3F;
            op[(2)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (3) != 3) {
            U64 const index = bits[(3)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(3)], (U16)entry);
            bits[(3)] <<= (entry >> 16) & 0x3F;
            op[(3)] += entry >> 24;
          }
        } while (0);
      } while (0);
      do {
        do {
          if (((0)) || (0) != 3) {
            U64 const index = bits[(0)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(0)], (U16)entry);
            bits[(0)] <<= (entry >> 16) & 0x3F;
            op[(0)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (1) != 3) {
            U64 const index = bits[(1)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(1)], (U16)entry);
            bits[(1)] <<= (entry >> 16) & 0x3F;
            op[(1)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (2) != 3) {
            U64 const index = bits[(2)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(2)], (U16)entry);
            bits[(2)] <<= (entry >> 16) & 0x3F;
            op[(2)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (3) != 3) {
            U64 const index = bits[(3)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(3)], (U16)entry);
            bits[(3)] <<= (entry >> 16) & 0x3F;
            op[(3)] += entry >> 24;
          }
        } while (0);
      } while (0);
      do {
        do {
          if (((0)) || (0) != 3) {
            U64 const index = bits[(0)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(0)], (U16)entry);
            bits[(0)] <<= (entry >> 16) & 0x3F;
            op[(0)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (1) != 3) {
            U64 const index = bits[(1)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(1)], (U16)entry);
            bits[(1)] <<= (entry >> 16) & 0x3F;
            op[(1)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (2) != 3) {
            U64 const index = bits[(2)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(2)], (U16)entry);
            bits[(2)] <<= (entry >> 16) & 0x3F;
            op[(2)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (3) != 3) {
            U64 const index = bits[(3)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(3)], (U16)entry);
            bits[(3)] <<= (entry >> 16) & 0x3F;
            op[(3)] += entry >> 24;
          }
        } while (0);
      } while (0);
      do {
        do {
          if (((0)) || (0) != 3) {
            U64 const index = bits[(0)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(0)], (U16)entry);
            bits[(0)] <<= (entry >> 16) & 0x3F;
            op[(0)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (1) != 3) {
            U64 const index = bits[(1)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(1)], (U16)entry);
            bits[(1)] <<= (entry >> 16) & 0x3F;
            op[(1)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (2) != 3) {
            U64 const index = bits[(2)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(2)], (U16)entry);
            bits[(2)] <<= (entry >> 16) & 0x3F;
            op[(2)] += entry >> 24;
          }
        } while (0);
        do {
          if (((0)) || (3) != 3) {
            U64 const index = bits[(3)] >> 53;
            size_t const entry = MEM_readLE32(&dtable[index]);
            MEM_write16(op[(3)], (U16)entry);
            bits[(3)] <<= (entry >> 16) & 0x3F;
            op[(3)] += entry >> 24;
          }
        } while (0);
      } while (0);

      do {
        if ((!0) || (3) != 3) {
          U64 const index = bits[(3)] >> 53;
          size_t const entry = MEM_readLE32(&dtable[index]);
          MEM_write16(op[(3)], (U16)entry);
          bits[(3)] <<= (entry >> 16) & 0x3F;
          op[(3)] += entry >> 24;
        }
      } while (0);

      do {
        do {
          if ((!0))
            do {
              if ((1) || (3) != 3) {
                U64 const index = bits[(3)] >> 53;
                size_t const entry = MEM_readLE32(&dtable[index]);
                MEM_write16(op[(3)], (U16)entry);
                bits[(3)] <<= (entry >> 16) & 0x3F;
                op[(3)] += entry >> 24;
              }
            } while (0);
          {
            U64 const ctz = ZSTD_countTrailingZeros64(bits[(0)]);
            U64 const nbBits = ctz & 7;
            U64 const nbBytes = ctz >> 3;
            ip[(0)] -= nbBytes;
            bits[(0)] = MEM_read64(ip[(0)]) | 1;
            bits[(0)] <<= nbBits;
          }
        } while (0);
        do {
          if ((!0))
            do {
              if ((1) || (3) != 3) {
                U64 const index = bits[(3)] >> 53;
                size_t const entry = MEM_readLE32(&dtable[index]);
                MEM_write16(op[(3)], (U16)entry);
                bits[(3)] <<= (entry >> 16) & 0x3F;
                op[(3)] += entry >> 24;
              }
            } while (0);
          {
            U64 const ctz = ZSTD_countTrailingZeros64(bits[(1)]);
            U64 const nbBits = ctz & 7;
            U64 const nbBytes = ctz >> 3;
            ip[(1)] -= nbBytes;
            bits[(1)] = MEM_read64(ip[(1)]) | 1;
            bits[(1)] <<= nbBits;
          }
        } while (0);
        do {
          if ((!0))
            do {
              if ((1) || (3) != 3) {
                U64 const index = bits[(3)] >> 53;
                size_t const entry = MEM_readLE32(&dtable[index]);
                MEM_write16(op[(3)], (U16)entry);
                bits[(3)] <<= (entry >> 16) & 0x3F;
                op[(3)] += entry >> 24;
              }
            } while (0);
          {
            U64 const ctz = ZSTD_countTrailingZeros64(bits[(2)]);
            U64 const nbBits = ctz & 7;
            U64 const nbBytes = ctz >> 3;
            ip[(2)] -= nbBytes;
            bits[(2)] = MEM_read64(ip[(2)]) | 1;
            bits[(2)] <<= nbBits;
          }
        } while (0);
        do {
          if ((!0))
            do {
              if ((1) || (3) != 3) {
                U64 const index = bits[(3)] >> 53;
                size_t const entry = MEM_readLE32(&dtable[index]);
                MEM_write16(op[(3)], (U16)entry);
                bits[(3)] <<= (entry >> 16) & 0x3F;
                op[(3)] += entry >> 24;
              }
            } while (0);
          {
            U64 const ctz = ZSTD_countTrailingZeros64(bits[(3)]);
            U64 const nbBits = ctz & 7;
            U64 const nbBytes = ctz >> 3;
            ip[(3)] -= nbBytes;
            bits[(3)] = MEM_read64(ip[(3)]) | 1;
            bits[(3)] <<= nbBits;
          }
        } while (0);
      } while (0);
    } while (op[3] < olimit);
  }

_out:

  __builtin_memcpy((&args->bits), (&bits), (sizeof(bits)));
  __builtin_memcpy(((void *)(&args->ip)), (&ip), (sizeof(ip)));
  __builtin_memcpy((&args->op), (&op), (sizeof(op)));
}

DEVICE_INLINE size_t HUF_decompress4X2_usingDTable_internal_fast(
    void *dst, size_t dstSize, const void *cSrc, size_t cSrcSize,
    const HUF_DTable *DTable, HUF_DecompressFastLoopFn loopFn) {
  void const *dt = DTable + 1;
  const BYTE *const ilowest = (const BYTE *)cSrc;
  BYTE *const oend = (BYTE *)ZSTD_maybeNullPtrAdd(dst, (ptrdiff_t)dstSize);
  HUF_DecompressFastArgs args;
  {
    size_t const ret = HUF_DecompressFastArgs_init(&args, dst, dstSize, cSrc,
                                                   cSrcSize, DTable);
    do {
      size_t const err_code = (ret);
      if (ERR_isError(err_code)) {
        do {
        } while (0);
        do {
          if (0) {
            _force_has_format_string("Failed to init asm args");
          }
        } while (0);
        do {
        } while (0);
        do {
        } while (0);
        return err_code;
      }
    } while (0);
    if (ret == 0)
      return 0;
  }

  loopFn(&args);

  (void)ilowest;

  {
    size_t const segmentSize = (dstSize + 3) / 4;
    BYTE *segmentEnd = (BYTE *)dst;
    int i;
    for (i = 0; i < 4; ++i) {
      BIT_DStream_t bit;
      if (segmentSize <= (size_t)(oend - segmentEnd))
        segmentEnd += segmentSize;
      else
        segmentEnd = oend;
      do {
        size_t const err_code =
            (HUF_initRemainingDStream(&bit, &args, i, segmentEnd));
        if (ERR_isError(err_code)) {
          do {
          } while (0);
          do {
            if (0) {
              _force_has_format_string("corruption");
            }
          } while (0);
          do {
          } while (0);
          do {
          } while (0);
          return err_code;
        }
      } while (0);
      args.op[i] += HUF_decodeStreamX2(args.op[i], &bit, segmentEnd,
                                       (HUF_DEltX2 const *)dt, 11);
      if (args.op[i] != segmentEnd)
        return ((size_t)-ZSTD_error_corruption_detected);
    }
  }

  return dstSize;
}

DEVICE_INLINE size_t HUF_decompress4X2_usingDTable_internal(
    void *dst, size_t dstSize, void const *cSrc, size_t cSrcSize,
    HUF_DTable const *DTable, int flags) {
  HUF_DecompressUsingDTableFn fallbackFn =
      HUF_decompress4X2_usingDTable_internal_default;
  HUF_DecompressFastLoopFn loopFn =
      HUF_decompress4X2_usingDTable_internal_fast_c_loop;

  if (flags & HUF_flags_bmi2) {
    fallbackFn = HUF_decompress4X2_usingDTable_internal_bmi2;

  } else {
    return fallbackFn(dst, dstSize, cSrc, cSrcSize, DTable);
  }
  if (1 && !(flags & HUF_flags_disableFast)) {
    size_t const ret = HUF_decompress4X2_usingDTable_internal_fast(
        dst, dstSize, cSrc, cSrcSize, DTable, loopFn);
    if (ret != 0)
      return ret;
  }
  return fallbackFn(dst, dstSize, cSrc, cSrcSize, DTable);
}

DEVICE_INLINE size_t HUF_decompress1X2_usingDTable_internal_default(
    void *dst, size_t dstSize, const void *cSrc, size_t cSrcSize,
    const HUF_DTable *DTable) {
  return HUF_decompress1X2_usingDTable_internal_body(dst, dstSize, cSrc,
                                                     cSrcSize, DTable);
}
DEVICE_INLINE size_t HUF_decompress1X2_usingDTable_internal_bmi2(
    void *dst, size_t dstSize, const void *cSrc, size_t cSrcSize,
    const HUF_DTable *DTable) {
  return HUF_decompress1X2_usingDTable_internal_body(dst, dstSize, cSrc,
                                                     cSrcSize, DTable);
}
DEVICE_INLINE size_t HUF_decompress1X2_usingDTable_internal(
    void *dst, size_t dstSize, void const *cSrc, size_t cSrcSize,
    HUF_DTable const *DTable, int flags) {
  if (flags & HUF_flags_bmi2) {
    return HUF_decompress1X2_usingDTable_internal_bmi2(dst, dstSize, cSrc,
                                                       cSrcSize, DTable);
  }
  return HUF_decompress1X2_usingDTable_internal_default(dst, dstSize, cSrc,
                                                        cSrcSize, DTable);
}

DEVICE_INLINE size_t HUF_decompress1X2_DCtx_wksp(
    HUF_DTable *DCtx, void *dst, size_t dstSize, const void *cSrc,
    size_t cSrcSize, void *workSpace, size_t wkspSize, int flags) {
  const BYTE *ip = (const BYTE *)cSrc;

  size_t const hSize =
      HUF_readDTableX2_wksp(DCtx, cSrc, cSrcSize, workSpace, wkspSize, flags);
  if (ERR_isError(hSize))
    return hSize;
  if (hSize >= cSrcSize)
    return ((size_t)-ZSTD_error_srcSize_wrong);
  ip += hSize;
  cSrcSize -= hSize;

  return HUF_decompress1X2_usingDTable_internal(dst, dstSize, ip, cSrcSize,
                                                DCtx, flags);
}

DEVICE_INLINE size_t HUF_decompress4X2_DCtx_wksp(
    HUF_DTable *dctx, void *dst, size_t dstSize, const void *cSrc,
    size_t cSrcSize, void *workSpace, size_t wkspSize, int flags) {
  const BYTE *ip = (const BYTE *)cSrc;

  size_t hSize =
      HUF_readDTableX2_wksp(dctx, cSrc, cSrcSize, workSpace, wkspSize, flags);
  if (ERR_isError(hSize))
    return hSize;
  if (hSize >= cSrcSize)
    return ((size_t)-ZSTD_error_srcSize_wrong);
  ip += hSize;
  cSrcSize -= hSize;

  return HUF_decompress4X2_usingDTable_internal(dst, dstSize, ip, cSrcSize,
                                                dctx, flags);
}

typedef struct {
  U32 tableTime;
  U32 decode256Time;
} algo_time_t;

static const DEVICE_CONSTANT algo_time_t
    algoTime[16 /* Quantization */][2 /* single, double */] = {
        /* single, double, quad */
        {{0, 0}, {1, 1}},           /* Q==0 : impossible */
        {{0, 0}, {1, 1}},           /* Q==1 : impossible */
        {{150, 216}, {381, 119}},   /* Q == 2 : 12-18% */
        {{170, 205}, {514, 112}},   /* Q == 3 : 18-25% */
        {{177, 199}, {539, 110}},   /* Q == 4 : 25-32% */
        {{197, 194}, {644, 107}},   /* Q == 5 : 32-38% */
        {{221, 192}, {735, 107}},   /* Q == 6 : 38-44% */
        {{256, 189}, {881, 106}},   /* Q == 7 : 44-50% */
        {{359, 188}, {1167, 109}},  /* Q == 8 : 50-56% */
        {{582, 187}, {1570, 114}},  /* Q == 9 : 56-62% */
        {{688, 187}, {1712, 122}},  /* Q ==10 : 62-69% */
        {{825, 186}, {1965, 136}},  /* Q ==11 : 69-75% */
        {{976, 185}, {2131, 150}},  /* Q ==12 : 75-81% */
        {{1180, 186}, {2070, 175}}, /* Q ==13 : 81-87% */
        {{1377, 185}, {1731, 202}}, /* Q ==14 : 87-93% */
        {{1412, 185}, {1695, 202}}, /* Q ==15 : 93-99% */
};

/*
 * HUF_decompress() does the following:
 * 1. select the decompression algorithm (X1, X2) based on pre-computed
 * heuristics
 * 2. build Huffman table from save, using HUF_readDTableX?()
 * 3. decode 1 or 4 segments in parallel using HUF_decompress?X?_usingDTable()
 */

/** HUF_selectDecoder() :
 *  Tells which decoder is likely to decode faster,
 *  based on a set of pre-computed metrics.
 * @return : 0==HUF_decompress4X1, 1==HUF_decompress4X2 .
 *  Assumption : 0 < dstSize <= 128 KB */
DEVICE_INLINE U32 HUF_selectDecoder(size_t dstSize, size_t cSrcSize) {

  {
    U32 const Q = (cSrcSize >= dstSize) ? 15 : (U32)(cSrcSize * 16 / dstSize);
    U32 const D256 = (U32)(dstSize >> 8);
    U32 const DTime0 =
        algoTime[Q][0].tableTime + (algoTime[Q][0].decode256Time * D256);
    U32 DTime1 =
        algoTime[Q][1].tableTime + (algoTime[Q][1].decode256Time * D256);
    DTime1 += DTime1 >> 5;
    return DTime1 < DTime0;
  }
}

// forward-declare

DEVICE_INLINE size_t HUF_decompress1X1_DCtx_wksp(
    HUF_DTable *dctx, void *dst, size_t dstSize, const void *cSrc,
    size_t cSrcSize, void *workSpace, size_t wkspSize, int flags);

DEVICE_INLINE size_t HUF_decompress1X_DCtx_wksp(
    HUF_DTable *dctx, void *dst, size_t dstSize, const void *cSrc,
    size_t cSrcSize, void *workSpace, size_t wkspSize, int flags) {

  if (dstSize == 0)
    return ((size_t)-ZSTD_error_dstSize_tooSmall);
  if (cSrcSize > dstSize)
    return ((size_t)-ZSTD_error_corruption_detected);
  if (cSrcSize == dstSize) {
    __builtin_memcpy((dst), (cSrc), (dstSize));
    return dstSize;
  }
  if (cSrcSize == 1) {
    __builtin_memset((dst), (*(const BYTE *)cSrc), (dstSize));
    return dstSize;
  }

  {
    U32 const algoNb = HUF_selectDecoder(dstSize, cSrcSize);
    return algoNb
               ? HUF_decompress1X2_DCtx_wksp(dctx, dst, dstSize, cSrc, cSrcSize,
                                             workSpace, wkspSize, flags)
               : HUF_decompress1X1_DCtx_wksp(dctx, dst, dstSize, cSrc, cSrcSize,
                                             workSpace, wkspSize, flags);
  }
}

DEVICE_INLINE size_t HUF_decompress1X_usingDTable(void *dst, size_t maxDstSize,
                                                  const void *cSrc,
                                                  size_t cSrcSize,
                                                  const HUF_DTable *DTable,
                                                  int flags) {
  DTableDesc const dtd = HUF_getDTableDesc(DTable);
  return dtd.tableType
             ? HUF_decompress1X2_usingDTable_internal(dst, maxDstSize, cSrc,
                                                      cSrcSize, DTable, flags)
             : HUF_decompress1X1_usingDTable_internal(dst, maxDstSize, cSrc,
                                                      cSrcSize, DTable, flags);
}

DEVICE_INLINE size_t HUF_decompress1X1_DCtx_wksp(
    HUF_DTable *dctx, void *dst, size_t dstSize, const void *cSrc,
    size_t cSrcSize, void *workSpace, size_t wkspSize, int flags) {
  const BYTE *ip = (const BYTE *)cSrc;

  size_t const hSize =
      HUF_readDTableX1_wksp(dctx, cSrc, cSrcSize, workSpace, wkspSize, flags);
  if (ERR_isError(hSize))
    return hSize;
  if (hSize >= cSrcSize)
    return ((size_t)-ZSTD_error_srcSize_wrong);
  ip += hSize;
  cSrcSize -= hSize;

  return HUF_decompress1X1_usingDTable_internal(dst, dstSize, ip, cSrcSize,
                                                dctx, flags);
}

DEVICE_INLINE size_t HUF_decompress4X_usingDTable(void *dst, size_t maxDstSize,
                                                  const void *cSrc,
                                                  size_t cSrcSize,
                                                  const HUF_DTable *DTable,
                                                  int flags) {
  DTableDesc const dtd = HUF_getDTableDesc(DTable);
  return dtd.tableType
             ? HUF_decompress4X2_usingDTable_internal(dst, maxDstSize, cSrc,
                                                      cSrcSize, DTable, flags)
             : HUF_decompress4X1_usingDTable_internal(dst, maxDstSize, cSrc,
                                                      cSrcSize, DTable, flags);
}

DEVICE_INLINE size_t HUF_decompress4X_hufOnly_wksp(
    HUF_DTable *dctx, void *dst, size_t dstSize, const void *cSrc,
    size_t cSrcSize, void *workSpace, size_t wkspSize, int flags) {

  if (dstSize == 0)
    return ((size_t)-ZSTD_error_dstSize_tooSmall);
  if (cSrcSize == 0)
    return ((size_t)-ZSTD_error_corruption_detected);

  {
    U32 const algoNb = HUF_selectDecoder(dstSize, cSrcSize);
    return algoNb
               ? HUF_decompress4X2_DCtx_wksp(dctx, dst, dstSize, cSrc, cSrcSize,
                                             workSpace, wkspSize, flags)
               : HUF_decompress4X1_DCtx_wksp(dctx, dst, dstSize, cSrc, cSrcSize,
                                             workSpace, wkspSize, flags);
  }
}

} // namespace zstd
} // namespace hipcomp
