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

#include "common.cuh"
#include "decompression_decode_block.cuh"
#include "decompression_decode_ddict.cuh"
#include "decompression_decode_fse.cuh"
#include "decompression_decode_huf.cuh"
#include "decompression_state.cuh"
#include "xxhash.cuh"

/* ***************************************************************
 *  Tuning parameters
 *****************************************************************/
/*!
 * HEAPMODE :
 * Select how default decompression function ZSTD_decompress() allocates its
 * context, on stack (0), or into heap (1, default; requires malloc()). Note
 * that functions with explicit context such as ZSTD_decompressDCtx() are
 * unaffected.
 */
#ifndef ZSTD_HEAPMODE
#define ZSTD_HEAPMODE 1
#endif
#undef ZSTD_HEAPMODE // NOTE(HIP/AMD): We do not consider heap mode

/*!
 *  LEGACY_SUPPORT :
 *  if set to 1+, ZSTD_decompress() can decode older formats (v0.1+)
 */
#ifndef ZSTD_LEGACY_SUPPORT
#define ZSTD_LEGACY_SUPPORT 0
#endif

/*!
 *  MAXWINDOWSIZE_DEFAULT :
 *  maximum window size accepted by DStream __by default__.
 *  Frames requiring more memory will be rejected.
 *  It's possible to set a different limit using ZSTD_DCtx_setMaxWindowSize().
 */
#ifndef ZSTD_MAXWINDOWSIZE_DEFAULT
#define ZSTD_MAXWINDOWSIZE_DEFAULT                                             \
  (((U32)1 << ZSTD_WINDOWLOG_LIMIT_DEFAULT) + 1)
#endif

/*!
 *  NO_FORWARD_PROGRESS_MAX :
 *  maximum allowed nb of calls to ZSTD_decompressStream()
 *  without any forward progress
 *  (defined as: no byte read from input, and no byte flushed to output)
 *  before triggering an error.
 */
#ifndef ZSTD_NO_FORWARD_PROGRESS_MAX
#define ZSTD_NO_FORWARD_PROGRESS_MAX 16
#endif

namespace hipcomp {
namespace zstd {

typedef struct ZSTD_CCtx_s ZSTD_CCtx;

// from zstd.h

typedef struct {
  size_t error;
  int lowerBound;
  int upperBound;
} ZSTD_bounds;

typedef enum {
  ZSTD_reset_session_only = 1,
  ZSTD_reset_parameters = 2,
  ZSTD_reset_session_and_parameters = 3
} ZSTD_ResetDirective;

typedef enum {
  ZSTD_d_windowLogMax =
      100, /* Select a size limit (in power of 2) beyond which
            * the streaming API will refuse to allocate memory buffer
            * in order to protect the host from unreasonable memory
            * requirements. This parameter is only useful in streaming mode,
            * since no internal buffer is allocated in single-pass mode. By
            * default, a decompression context accepts window sizes <= (1 <<
            * ZSTD_WINDOWLOG_LIMIT_DEFAULT). Special: value 0 means "use default
            * maximum windowLog". */

  /* note : additional experimental parameters are also available
   * within the experimental section of the API.
   * At the time of this writing, they include :
   * ZSTD_d_format
   * ZSTD_d_stableOutBuffer
   * ZSTD_d_forceIgnoreChecksum
   * ZSTD_d_refMultipleDDicts
   * ZSTD_d_disableHuffmanAssembly
   * ZSTD_d_maxBlockSize
   * Because they are not stable, it's necessary to define
   * ZSTD_STATIC_LINKING_ONLY to access them. note : never ever use
   * experimentalParam? names directly
   */
  ZSTD_d_experimentalParam1 = 1000,
  ZSTD_d_experimentalParam2 = 1001,
  ZSTD_d_experimentalParam3 = 1002,
  ZSTD_d_experimentalParam4 = 1003,
  ZSTD_d_experimentalParam5 = 1004,
  ZSTD_d_experimentalParam6 = 1005
} ZSTD_dParameter;

// end from zstd.h

/****************************
 *  Streaming
 ****************************/

typedef struct ZSTD_inBuffer_s {
  const void *src; /**< start of input buffer */
  size_t size;     /**< size of input buffer */
  size_t pos; /**< position where reading stopped. Will be updated. Necessarily
                 0 <= pos <= size */
} ZSTD_inBuffer;

typedef ZSTD_DCtx ZSTD_DStream;

typedef struct ZSTD_DDict_s ZSTD_DDict;

typedef struct ZSTD_CCtx_params_s ZSTD_CCtx_params;

typedef enum {
  ZSTDnit_frameHeader,
  ZSTDnit_blockHeader,
  ZSTDnit_block,
  ZSTDnit_lastBlock,
  ZSTDnit_checksum,
  ZSTDnit_skippableFrame
} ZSTD_nextInputType_e;

// from lib/common/zstd_internal.h

/*-*******************************************
 *  Private declarations
 *********************************************/

/**
 * Contains the compressed frame size and an upper-bound for the decompressed
 * frame size. Note: before using `compressedSize`, check for errors using
 * ZSTD_isError(). similarly, before using `decompressedBound`, check for errors
 * using: `decompressedBound != ZSTD_CONTENTSIZE_ERROR`
 */
typedef struct {
  size_t nbBlocks;
  size_t compressedSize;
  unsigned long long decompressedBound;
} ZSTD_frameSizeInfo; /* decompress & legacy */

// end from lib/common/zstd_internal.h

#if defined(ZSTD_LEGACY_SUPPORT) && (ZSTD_LEGACY_SUPPORT >= 1)
#error HIP/AMD: legacy versions of ZSTD are not supported

// from lib/legacy/*.h

typedef struct ZSTDv05_DCtx_s ZSTDv05_DCtx;

typedef enum {
  ZSTDv05_fast,
  ZSTDv05_greedy,
  ZSTDv05_lazy,
  ZSTDv05_lazy2,
  ZSTDv05_btlazy2,
  ZSTDv05_opt,
  ZSTDv05_btopt
} ZSTDv05_strategy;

typedef struct {
  U64 srcSize;
  U32 windowLog;
  U32 contentLog;
  U32 hashLog;
  U32 searchLog;
  U32 searchLength;
  U32 targetLength;
  ZSTDv05_strategy strategy;
} ZSTDv05_parameters;

typedef struct ZBUFFv05_DCtx_s ZBUFFv05_DCtx;

typedef struct ZSTDv06_DCtx_s ZSTDv06_DCtx;

struct ZSTDv06_frameParams_s {
  unsigned long long frameContentSize;
  unsigned windowLog;
};

typedef struct ZSTDv06_frameParams_s ZSTDv06_frameParams;

typedef struct ZBUFFv06_DCtx_s ZBUFFv06_DCtx;

typedef struct ZSTDv07_DCtx_s ZSTDv07_DCtx;

typedef struct {
  unsigned long long frameContentSize;
  unsigned windowSize;
  unsigned dictID;
  unsigned checksumFlag;
} ZSTDv07_frameParams;

typedef struct ZBUFFv07_DCtx_s ZBUFFv07_DCtx;

#define ZSTDv01_magicNumber 0xFD2FB51E   /* Big Endian version */
#define ZSTDv01_magicNumberLE 0x1EB52FFD /* Little Endian version */
#define ZSTDv02_magicNumber 0xFD2FB522   /* v0.2 */
#define ZSTDv03_magicNumber 0xFD2FB523   /* v0.3 */
#define ZSTDv04_magicNumber 0xFD2FB524   /* v0.4 */
#define ZSTDv05_MAGICNUMBER 0xFD2FB525   /* v0.5 */
#define ZSTDv06_MAGICNUMBER 0xFD2FB526   /* v0.6 */
#define ZSTDv07_MAGICNUMBER 0xFD2FB527   /* v0.7 */

/** ZSTD_isLegacy() :
    @return : > 0 if supported by legacy decoder. 0 otherwise.
              return value is the version.
*/

DEVICE_INLINE unsigned ZSTD_isLegacy(const void *src, size_t srcSize) {
  U32 magicNumberLE;
  if (srcSize < 4)
    return 0;
  magicNumberLE = MEM_readLE32(src);
  switch (magicNumberLE) {
#if (ZSTD_LEGACY_SUPPORT <= 1)
  case ZSTDv01_magicNumberLE:
    return 1;
#endif
#if (ZSTD_LEGACY_SUPPORT <= 2)
  case ZSTDv02_magicNumber:
    return 2;
#endif
#if (ZSTD_LEGACY_SUPPORT <= 3)
  case ZSTDv03_magicNumber:
    return 3;
#endif
#if (ZSTD_LEGACY_SUPPORT <= 4)
  case ZSTDv04_magicNumber:
    return 4;
#endif
#if (ZSTD_LEGACY_SUPPORT <= 5)
  case ZSTDv05_MAGICNUMBER:
    return 5;
#endif
#if (ZSTD_LEGACY_SUPPORT <= 6)
  case ZSTDv06_MAGICNUMBER:
    return 6;
#endif
#if (ZSTD_LEGACY_SUPPORT <= 7)
  case ZSTDv07_MAGICNUMBER:
    return 7;
#endif
  default:
    return 0;
  }
}

DEVICE_INLINE unsigned long long
ZSTD_getDecompressedSize_legacy(const void *src, size_t srcSize) {
  U32 const version = ZSTD_isLegacy(src, srcSize);
  if (version < 5)
    return 0; /* no decompressed size in frame header, or not a legacy format */
#if (ZSTD_LEGACY_SUPPORT <= 5)
  if (version == 5) {
    ZSTDv05_parameters fParams;
    size_t const frResult = ZSTDv05_getFrameParams(&fParams, src, srcSize);
    if (frResult != 0)
      return 0;
    return fParams.srcSize;
  }
#endif
#if (ZSTD_LEGACY_SUPPORT <= 6)
  if (version == 6) {
    ZSTDv06_frameParams fParams;
    size_t const frResult = ZSTDv06_getFrameParams(&fParams, src, srcSize);
    if (frResult != 0)
      return 0;
    return fParams.frameContentSize;
  }
#endif
#if (ZSTD_LEGACY_SUPPORT <= 7)
  if (version == 7) {
    ZSTDv07_frameParams fParams;
    size_t const frResult = ZSTDv07_getFrameParams(&fParams, src, srcSize);
    if (frResult != 0)
      return 0;
    return fParams.frameContentSize;
  }
#endif
  return 0; /* should not be possible */
}

DEVICE_INLINE size_t ZSTD_decompressLegacy(void *dst, size_t dstCapacity,
                                           const void *src,
                                           size_t compressedSize,
                                           const void *dict, size_t dictSize) {
  U32 const version = ZSTD_isLegacy(src, compressedSize);
  char x;
  /* Avoid passing NULL to legacy decoding. */
  if (dst == NULL) {
    assert(dstCapacity == 0);
    dst = &x;
  }
  if (src == NULL) {
    assert(compressedSize == 0);
    src = &x;
  }
  if (dict == NULL) {
    assert(dictSize == 0);
    dict = &x;
  }
  (void)dst;
  (void)dstCapacity;
  (void)dict;
  (void)dictSize; /* unused when ZSTD_LEGACY_SUPPORT >= 8 */
  switch (version) {
#if (ZSTD_LEGACY_SUPPORT <= 1)
  case 1:
    return ZSTDv01_decompress(dst, dstCapacity, src, compressedSize);
#endif
#if (ZSTD_LEGACY_SUPPORT <= 2)
  case 2:
    return ZSTDv02_decompress(dst, dstCapacity, src, compressedSize);
#endif
#if (ZSTD_LEGACY_SUPPORT <= 3)
  case 3:
    return ZSTDv03_decompress(dst, dstCapacity, src, compressedSize);
#endif
#if (ZSTD_LEGACY_SUPPORT <= 4)
  case 4:
    return ZSTDv04_decompress(dst, dstCapacity, src, compressedSize);
#endif
#if (ZSTD_LEGACY_SUPPORT <= 5)
  case 5: {
    size_t result;
    ZSTDv05_DCtx *const zd = ZSTDv05_createDCtx();
    if (zd == NULL)
      return ERROR(memory_allocation);
    result = ZSTDv05_decompress_usingDict(zd, dst, dstCapacity, src,
                                          compressedSize, dict, dictSize);
    ZSTDv05_freeDCtx(zd);
    return result;
  }
#endif
#if (ZSTD_LEGACY_SUPPORT <= 6)
  case 6: {
    size_t result;
    ZSTDv06_DCtx *const zd = ZSTDv06_createDCtx();
    if (zd == NULL)
      return ERROR(memory_allocation);
    result = ZSTDv06_decompress_usingDict(zd, dst, dstCapacity, src,
                                          compressedSize, dict, dictSize);
    ZSTDv06_freeDCtx(zd);
    return result;
  }
#endif
#if (ZSTD_LEGACY_SUPPORT <= 7)
  case 7: {
    size_t result;
    ZSTDv07_DCtx *const zd = ZSTDv07_createDCtx();
    if (zd == NULL)
      return ERROR(memory_allocation);
    result = ZSTDv07_decompress_usingDict(zd, dst, dstCapacity, src,
                                          compressedSize, dict, dictSize);
    ZSTDv07_freeDCtx(zd);
    return result;
  }
#endif
  default:
    return ERROR(prefix_unknown);
  }
}

DEVICE_INLINE ZSTD_frameSizeInfo ZSTD_findFrameSizeInfoLegacy(const void *src,
                                                              size_t srcSize) {
  ZSTD_frameSizeInfo frameSizeInfo;
  U32 const version = ZSTD_isLegacy(src, srcSize);
  switch (version) {
#if (ZSTD_LEGACY_SUPPORT <= 1)
  case 1:
    ZSTDv01_findFrameSizeInfoLegacy(src, srcSize, &frameSizeInfo.compressedSize,
                                    &frameSizeInfo.decompressedBound);
    break;
#endif
#if (ZSTD_LEGACY_SUPPORT <= 2)
  case 2:
    ZSTDv02_findFrameSizeInfoLegacy(src, srcSize, &frameSizeInfo.compressedSize,
                                    &frameSizeInfo.decompressedBound);
    break;
#endif
#if (ZSTD_LEGACY_SUPPORT <= 3)
  case 3:
    ZSTDv03_findFrameSizeInfoLegacy(src, srcSize, &frameSizeInfo.compressedSize,
                                    &frameSizeInfo.decompressedBound);
    break;
#endif
#if (ZSTD_LEGACY_SUPPORT <= 4)
  case 4:
    ZSTDv04_findFrameSizeInfoLegacy(src, srcSize, &frameSizeInfo.compressedSize,
                                    &frameSizeInfo.decompressedBound);
    break;
#endif
#if (ZSTD_LEGACY_SUPPORT <= 5)
  case 5:
    ZSTDv05_findFrameSizeInfoLegacy(src, srcSize, &frameSizeInfo.compressedSize,
                                    &frameSizeInfo.decompressedBound);
    break;
#endif
#if (ZSTD_LEGACY_SUPPORT <= 6)
  case 6:
    ZSTDv06_findFrameSizeInfoLegacy(src, srcSize, &frameSizeInfo.compressedSize,
                                    &frameSizeInfo.decompressedBound);
    break;
#endif
#if (ZSTD_LEGACY_SUPPORT <= 7)
  case 7:
    ZSTDv07_findFrameSizeInfoLegacy(src, srcSize, &frameSizeInfo.compressedSize,
                                    &frameSizeInfo.decompressedBound);
    break;
#endif
  default:
    frameSizeInfo.compressedSize = ERROR(prefix_unknown);
    frameSizeInfo.decompressedBound = ZSTD_CONTENTSIZE_ERROR;
    break;
  }
  if (!ZSTD_isError(frameSizeInfo.compressedSize) &&
      frameSizeInfo.compressedSize > srcSize) {
    frameSizeInfo.compressedSize = ERROR(srcSize_wrong);
    frameSizeInfo.decompressedBound = ZSTD_CONTENTSIZE_ERROR;
  }
  /* In all cases, decompressedBound == nbBlocks * ZSTD_BLOCKSIZE_MAX.
   * So we can compute nbBlocks without having to change every function.
   */
  if (frameSizeInfo.decompressedBound != ZSTD_CONTENTSIZE_ERROR) {
    assert((frameSizeInfo.decompressedBound & (ZSTD_BLOCKSIZE_MAX - 1)) == 0);
    frameSizeInfo.nbBlocks =
        (size_t)(frameSizeInfo.decompressedBound / ZSTD_BLOCKSIZE_MAX);
  }
  return frameSizeInfo;
}

DEVICE_INLINE size_t ZSTD_findFrameCompressedSizeLegacy(const void *src,
                                                        size_t srcSize) {
  ZSTD_frameSizeInfo frameSizeInfo = ZSTD_findFrameSizeInfoLegacy(src, srcSize);
  return frameSizeInfo.compressedSize;
}

DEVICE_INLINE size_t ZSTD_freeLegacyStreamContext(void *legacyContext,
                                                  U32 version) {
  switch (version) {
  default:
  case 1:
  case 2:
  case 3:
    (void)legacyContext;
    return ERROR(version_unsupported);
#if (ZSTD_LEGACY_SUPPORT <= 4)
  case 4:
    return ZBUFFv04_freeDCtx((ZBUFFv04_DCtx *)legacyContext);
#endif
#if (ZSTD_LEGACY_SUPPORT <= 5)
  case 5:
    return ZBUFFv05_freeDCtx((ZBUFFv05_DCtx *)legacyContext);
#endif
#if (ZSTD_LEGACY_SUPPORT <= 6)
  case 6:
    return ZBUFFv06_freeDCtx((ZBUFFv06_DCtx *)legacyContext);
#endif
#if (ZSTD_LEGACY_SUPPORT <= 7)
  case 7:
    return ZBUFFv07_freeDCtx((ZBUFFv07_DCtx *)legacyContext);
#endif
  }
}

DEVICE_INLINE size_t ZSTD_initLegacyStream(void **legacyContext,
                                           U32 prevVersion, U32 newVersion,
                                           const void *dict, size_t dictSize) {
  char x;
  /* Avoid passing NULL to legacy decoding. */
  if (dict == NULL) {
    assert(dictSize == 0);
    dict = &x;
  }
  DEBUGLOG(5, "ZSTD_initLegacyStream for v0.%u", newVersion);
  if (prevVersion != newVersion)
    ZSTD_freeLegacyStreamContext(*legacyContext, prevVersion);
  switch (newVersion) {
  default:
  case 1:
  case 2:
  case 3:
    (void)dict;
    (void)dictSize;
    return 0;
#if (ZSTD_LEGACY_SUPPORT <= 4)
  case 4: {
    ZBUFFv04_DCtx *dctx = (prevVersion != newVersion)
                              ? ZBUFFv04_createDCtx()
                              : (ZBUFFv04_DCtx *)*legacyContext;
    if (dctx == NULL)
      return ERROR(memory_allocation);
    ZBUFFv04_decompressInit(dctx);
    ZBUFFv04_decompressWithDictionary(dctx, dict, dictSize);
    *legacyContext = dctx;
    return 0;
  }
#endif
#if (ZSTD_LEGACY_SUPPORT <= 5)
  case 5: {
    ZBUFFv05_DCtx *dctx = (prevVersion != newVersion)
                              ? ZBUFFv05_createDCtx()
                              : (ZBUFFv05_DCtx *)*legacyContext;
    if (dctx == NULL)
      return ERROR(memory_allocation);
    ZBUFFv05_decompressInitDictionary(dctx, dict, dictSize);
    *legacyContext = dctx;
    return 0;
  }
#endif
#if (ZSTD_LEGACY_SUPPORT <= 6)
  case 6: {
    ZBUFFv06_DCtx *dctx = (prevVersion != newVersion)
                              ? ZBUFFv06_createDCtx()
                              : (ZBUFFv06_DCtx *)*legacyContext;
    if (dctx == NULL)
      return ERROR(memory_allocation);
    ZBUFFv06_decompressInitDictionary(dctx, dict, dictSize);
    *legacyContext = dctx;
    return 0;
  }
#endif
#if (ZSTD_LEGACY_SUPPORT <= 7)
  case 7: {
    ZBUFFv07_DCtx *dctx = (prevVersion != newVersion)
                              ? ZBUFFv07_createDCtx()
                              : (ZBUFFv07_DCtx *)*legacyContext;
    if (dctx == NULL)
      return ERROR(memory_allocation);
    ZBUFFv07_decompressInitDictionary(dctx, dict, dictSize);
    *legacyContext = dctx;
    return 0;
  }
#endif
  }
}

DEVICE_INLINE size_t ZSTD_decompressLegacyStream(void *legacyContext,
                                                 U32 version,
                                                 ZSTD_outBuffer *output,
                                                 ZSTD_inBuffer *input) {
  static char x;
  /* Avoid passing NULL to legacy decoding. */
  if (output->dst == NULL) {
    assert(output->size == 0);
    output->dst = &x;
  }
  if (input->src == NULL) {
    assert(input->size == 0);
    input->src = &x;
  }
  DEBUGLOG(5, "ZSTD_decompressLegacyStream for v0.%u", version);
  switch (version) {
  default:
  case 1:
  case 2:
  case 3:
    (void)legacyContext;
    (void)output;
    (void)input;
    return ERROR(version_unsupported);
#if (ZSTD_LEGACY_SUPPORT <= 4)
  case 4: {
    ZBUFFv04_DCtx *dctx = (ZBUFFv04_DCtx *)legacyContext;
    const void *src = (const char *)input->src + input->pos;
    size_t readSize = input->size - input->pos;
    void *dst = (char *)output->dst + output->pos;
    size_t decodedSize = output->size - output->pos;
    size_t const hintSize =
        ZBUFFv04_decompressContinue(dctx, dst, &decodedSize, src, &readSize);
    output->pos += decodedSize;
    input->pos += readSize;
    return hintSize;
  }
#endif
#if (ZSTD_LEGACY_SUPPORT <= 5)
  case 5: {
    ZBUFFv05_DCtx *dctx = (ZBUFFv05_DCtx *)legacyContext;
    const void *src = (const char *)input->src + input->pos;
    size_t readSize = input->size - input->pos;
    void *dst = (char *)output->dst + output->pos;
    size_t decodedSize = output->size - output->pos;
    size_t const hintSize =
        ZBUFFv05_decompressContinue(dctx, dst, &decodedSize, src, &readSize);
    output->pos += decodedSize;
    input->pos += readSize;
    return hintSize;
  }
#endif
#if (ZSTD_LEGACY_SUPPORT <= 6)
  case 6: {
    ZBUFFv06_DCtx *dctx = (ZBUFFv06_DCtx *)legacyContext;
    const void *src = (const char *)input->src + input->pos;
    size_t readSize = input->size - input->pos;
    void *dst = (char *)output->dst + output->pos;
    size_t decodedSize = output->size - output->pos;
    size_t const hintSize =
        ZBUFFv06_decompressContinue(dctx, dst, &decodedSize, src, &readSize);
    output->pos += decodedSize;
    input->pos += readSize;
    return hintSize;
  }
#endif
#if (ZSTD_LEGACY_SUPPORT <= 7)
  case 7: {
    ZBUFFv07_DCtx *dctx = (ZBUFFv07_DCtx *)legacyContext;
    const void *src = (const char *)input->src + input->pos;
    size_t readSize = input->size - input->pos;
    void *dst = (char *)output->dst + output->pos;
    size_t decodedSize = output->size - output->pos;
    size_t const hintSize =
        ZBUFFv07_decompressContinue(dctx, dst, &decodedSize, src, &readSize);
    output->pos += decodedSize;
    input->pos += readSize;
    return hintSize;
  }
#endif
  }
}

#endif // defined(ZSTD_LEGACY_SUPPORT) && (ZSTD_LEGACY_SUPPORT >= 1)

/*************************************
 * Multiple DDicts Hashset internals *
 *************************************/

#define DDICT_HASHSET_MAX_LOAD_FACTOR_COUNT_MULT 4
#define DDICT_HASHSET_MAX_LOAD_FACTOR_SIZE_MULT                                \
  3 /* These two constants represent SIZE_MULT/COUNT_MULT load factor without  \
     * using a float. Currently, that means a 0.75 load factor. So, if count * \
     * COUNT_MULT / size * SIZE_MULT != 0, then we've exceeded the load factor \
     * of the ddict hash set.                                                  \
     */

#define DDICT_HASHSET_TABLE_BASE_SIZE 64
#define DDICT_HASHSET_RESIZE_FACTOR 2

/* Hash function to determine starting position of dict insertion within the
 * table Returns an index between [0, hashSet->ddictPtrTableSize]
 */
DEVICE_INLINE size_t
ZSTD_DDictHashSet_getIndex(const ZSTD_DDictHashSet *hashSet, U32 dictID) {
  const U64 hash = XXH64(&dictID, sizeof(U32), 0);
  /* DDict ptr table size is a multiple of 2, use size - 1 as mask to get index
   * within [0, hashSet->ddictPtrTableSize) */
  return hash & (hashSet->ddictPtrTableSize - 1);
}

/* Adds DDict to a hashset without resizing it.
 * If inserting a DDict with a dictID that already exists in the set, replaces
 * the one in the set. Returns 0 if successful, or a zstd error code if
 * something went wrong.
 */
DEVICE_INLINE size_t ZSTD_DDictHashSet_emplaceDDict(ZSTD_DDictHashSet *hashSet,
                                                    const ZSTD_DDict *ddict) {
  const U32 dictID = ZSTD_getDictID_fromDDict(ddict);
  size_t idx = ZSTD_DDictHashSet_getIndex(hashSet, dictID);
  const size_t idxRangeMask = hashSet->ddictPtrTableSize - 1;
  RETURN_ERROR_IF(hashSet->ddictPtrCount == hashSet->ddictPtrTableSize, GENERIC,
                  "Hash set is full!");
  DEBUGLOG(4, "Hashed index: for dictID: %u is %zu", dictID, idx);
  while (hashSet->ddictPtrTable[idx] != NULL) {
    /* Replace existing ddict if inserting ddict with same dictID */
    if (ZSTD_getDictID_fromDDict(hashSet->ddictPtrTable[idx]) == dictID) {
      DEBUGLOG(4, "DictID already exists, replacing rather than adding");
      hashSet->ddictPtrTable[idx] = ddict;
      return 0;
    }
    idx &= idxRangeMask;
    idx++;
  }
  DEBUGLOG(4, "Final idx after probing for dictID %u is: %zu", dictID, idx);
  hashSet->ddictPtrTable[idx] = ddict;
  hashSet->ddictPtrCount++;
  return 0;
}

/* Expands hash table by factor of DDICT_HASHSET_RESIZE_FACTOR and
 * rehashes all values, allocates new table, frees old table.
 * Returns 0 on success, otherwise a zstd error code.
 */
DEVICE_INLINE size_t ZSTD_DDictHashSet_expand(ZSTD_DDictHashSet *hashSet,
                                              ZSTD_customMem customMem) {
  size_t newTableSize =
      hashSet->ddictPtrTableSize * DDICT_HASHSET_RESIZE_FACTOR;
  const ZSTD_DDict **newTable = (const ZSTD_DDict **)ZSTD_customCalloc(
      sizeof(ZSTD_DDict *) * newTableSize, customMem);
  const ZSTD_DDict **oldTable = hashSet->ddictPtrTable;
  size_t oldTableSize = hashSet->ddictPtrTableSize;
  size_t i;

  DEBUGLOG(4, "Expanding DDict hash table! Old size: %zu new size: %zu",
           oldTableSize, newTableSize);
  RETURN_ERROR_IF(!newTable, memory_allocation,
                  "Expanded hashset allocation failed!");
  hashSet->ddictPtrTable = newTable;
  hashSet->ddictPtrTableSize = newTableSize;
  hashSet->ddictPtrCount = 0;
  for (i = 0; i < oldTableSize; ++i) {
    if (oldTable[i] != NULL) {
      FORWARD_IF_ERROR(ZSTD_DDictHashSet_emplaceDDict(hashSet, oldTable[i]),
                       "");
    }
  }
  ZSTD_customFree((void *)oldTable, customMem);
  DEBUGLOG(4, "Finished re-hash");
  return 0;
}

/* Fetches a DDict with the given dictID
 * Returns the ZSTD_DDict* with the requested dictID. If it doesn't exist, then
 * returns NULL.
 */
DEVICE_INLINE const ZSTD_DDict *
ZSTD_DDictHashSet_getDDict(ZSTD_DDictHashSet *hashSet, U32 dictID) {
  size_t idx = ZSTD_DDictHashSet_getIndex(hashSet, dictID);
  const size_t idxRangeMask = hashSet->ddictPtrTableSize - 1;
  DEBUGLOG(4, "Hashed index: for dictID: %u is %zu", dictID, idx);
  for (;;) {
    size_t currDictID = ZSTD_getDictID_fromDDict(hashSet->ddictPtrTable[idx]);
    if (currDictID == dictID || currDictID == 0) {
      /* currDictID == 0 implies a NULL ddict entry */
      break;
    } else {
      idx &= idxRangeMask; /* Goes to start of table when we reach the end */
      idx++;
    }
  }
  DEBUGLOG(4, "Final idx after probing for dictID %u is: %zu", dictID, idx);
  return hashSet->ddictPtrTable[idx];
}

/* Allocates space for and returns a ddict hash set
 * The hash set's ZSTD_DDict* table has all values automatically set to NULL to
 * begin with. Returns NULL if allocation failed.
 */
DEVICE_INLINE ZSTD_DDictHashSet *
ZSTD_createDDictHashSet(ZSTD_customMem customMem) {
  ZSTD_DDictHashSet *ret = (ZSTD_DDictHashSet *)ZSTD_customMalloc(
      sizeof(ZSTD_DDictHashSet), customMem);
  DEBUGLOG(4, "Allocating new hash set");
  if (!ret)
    return NULL;
  ret->ddictPtrTable = (const ZSTD_DDict **)ZSTD_customCalloc(
      DDICT_HASHSET_TABLE_BASE_SIZE * sizeof(ZSTD_DDict *), customMem);
  if (!ret->ddictPtrTable) {
    ZSTD_customFree(ret, customMem);
    return NULL;
  }
  ret->ddictPtrTableSize = DDICT_HASHSET_TABLE_BASE_SIZE;
  ret->ddictPtrCount = 0;
  return ret;
}

/* Frees the table of ZSTD_DDict* within a hashset, then frees the hashset
 * itself. Note: The ZSTD_DDict* within the table are NOT freed.
 */
DEVICE_INLINE void ZSTD_freeDDictHashSet(ZSTD_DDictHashSet *hashSet,
                                         ZSTD_customMem customMem) {
  DEBUGLOG(4, "Freeing ddict hash set");
  if (hashSet && hashSet->ddictPtrTable) {
    ZSTD_customFree((void *)hashSet->ddictPtrTable, customMem);
  }
  if (hashSet) {
    ZSTD_customFree(hashSet, customMem);
  }
}

/* Public function: Adds a DDict into the ZSTD_DDictHashSet, possibly triggering
 * a resize of the hash set. Returns 0 on success, or a ZSTD error.
 */
DEVICE_INLINE size_t ZSTD_DDictHashSet_addDDict(ZSTD_DDictHashSet *hashSet,
                                                const ZSTD_DDict *ddict,
                                                ZSTD_customMem customMem) {
  DEBUGLOG(4, "Adding dict ID: %u to hashset with - Count: %zu Tablesize: %zu",
           ZSTD_getDictID_fromDDict(ddict), hashSet->ddictPtrCount,
           hashSet->ddictPtrTableSize);
  if (hashSet->ddictPtrCount * DDICT_HASHSET_MAX_LOAD_FACTOR_COUNT_MULT /
          hashSet->ddictPtrTableSize *
          DDICT_HASHSET_MAX_LOAD_FACTOR_SIZE_MULT !=
      0) {
    FORWARD_IF_ERROR(ZSTD_DDictHashSet_expand(hashSet, customMem), "");
  }
  FORWARD_IF_ERROR(ZSTD_DDictHashSet_emplaceDDict(hashSet, ddict), "");
  return 0;
}

/*-*************************************************************
 *   Context management
 ***************************************************************/

DEVICE_INLINE size_t ZSTD_sizeof_DCtx(const ZSTD_DCtx *dctx) {
  if (dctx == NULL)
    return 0; /* support sizeof NULL */
  return sizeof(*dctx) + ZSTD_sizeof_DDict(dctx->ddictLocal) +
         dctx->inBuffSize + dctx->outBuffSize;
}

DEVICE_INLINE size_t ZSTD_estimateDCtxSize(void) { return sizeof(ZSTD_DCtx); }

DEVICE_INLINE size_t ZSTD_startingInputLength(ZSTD_format_e format) {
  size_t const startingInputLength = ZSTD_FRAMEHEADERSIZE_PREFIX(format);
  /* only supports formats ZSTD_f_zstd1 and ZSTD_f_zstd1_magicless */
  assert((format == ZSTD_f_zstd1) || (format == ZSTD_f_zstd1_magicless));
  return startingInputLength;
}

DEVICE_INLINE void ZSTD_DCtx_resetParameters(ZSTD_DCtx *dctx) {
  dctx->format = ZSTD_f_zstd1;
  dctx->maxWindowSize = ZSTD_MAXWINDOWSIZE_DEFAULT;
  dctx->outBufferMode = ZSTD_bm_buffered;
  dctx->forceIgnoreChecksum = ZSTD_d_validateChecksum;
  dctx->refMultipleDDicts = ZSTD_rmd_refSingleDDict;
  dctx->disableHufAsm = 0;
  dctx->maxBlockSizeParam = 0;
}

DEVICE_INLINE void ZSTD_initDCtx_internal(ZSTD_DCtx *dctx) {
  dctx->staticSize = 0;
  dctx->ddict = NULL;
  dctx->ddictLocal = NULL;
  dctx->dictEnd = NULL;
  dctx->ddictIsCold = 0;
  dctx->dictUses = ZSTD_dont_use;
  dctx->inBuff = NULL;
  dctx->inBuffSize = 0;
  dctx->outBuffSize = 0;
#if defined(ZSTD_LEGACY_SUPPORT) && (ZSTD_LEGACY_SUPPORT >= 1)
  dctx->legacyContext = NULL;
  dctx->previousLegacyVersion = 0;
#endif
  dctx->noForwardProgress = 0;
  dctx->isFrameDecompression = 1;
#if DYNAMIC_BMI2
  dctx->bmi2 = ZSTD_cpuSupportsBmi2();
#endif
  dctx->ddictSet = NULL;
  ZSTD_DCtx_resetParameters(dctx);
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  dctx->dictContentEndForFuzzing = NULL;
#endif
}

DEVICE_INLINE ZSTD_DCtx *ZSTD_initStaticDCtx(void *workspace,
                                             size_t workspaceSize) {
  ZSTD_DCtx *const dctx = (ZSTD_DCtx *)workspace;

  if ((size_t)workspace & 7)
    return NULL; /* 8-aligned */
  if (workspaceSize < sizeof(ZSTD_DCtx))
    return NULL; /* minimum size */

  ZSTD_initDCtx_internal(dctx);
  dctx->staticSize = workspaceSize;
  dctx->inBuff = (char *)(dctx + 1);
  return dctx;
}

DEVICE_INLINE ZSTD_DCtx *ZSTD_createDCtx_internal(ZSTD_customMem customMem) {
  if ((!customMem.customAlloc) ^ (!customMem.customFree))
    return NULL;

  {
    ZSTD_DCtx *const dctx =
        (ZSTD_DCtx *)ZSTD_customMalloc(sizeof(*dctx), customMem);
    if (!dctx)
      return NULL;
    dctx->customMem = customMem;
    ZSTD_initDCtx_internal(dctx);
    return dctx;
  }
}

DEVICE_INLINE ZSTD_DCtx *ZSTD_createDCtx_advanced(ZSTD_customMem customMem) {
  return ZSTD_createDCtx_internal(customMem);
}

DEVICE_INLINE ZSTD_DCtx *ZSTD_createDCtx(void) {
  DEBUGLOG(3, "ZSTD_createDCtx");
  return ZSTD_createDCtx_internal(ZSTD_defaultCMem);
}

DEVICE_INLINE void ZSTD_clearDict(ZSTD_DCtx *dctx) {
  ZSTD_freeDDict(dctx->ddictLocal);
  dctx->ddictLocal = NULL;
  dctx->ddict = NULL;
  dctx->dictUses = ZSTD_dont_use;
}

DEVICE_INLINE size_t ZSTD_freeDCtx(ZSTD_DCtx *dctx) {
  if (dctx == NULL)
    return 0; /* support free on NULL */
  RETURN_ERROR_IF(dctx->staticSize, memory_allocation,
                  "not compatible with static DCtx");
  {
    ZSTD_customMem const cMem = dctx->customMem;
    ZSTD_clearDict(dctx);
    ZSTD_customFree(dctx->inBuff, cMem);
    dctx->inBuff = NULL;
#if defined(ZSTD_LEGACY_SUPPORT) && (ZSTD_LEGACY_SUPPORT >= 1)
    if (dctx->legacyContext)
      ZSTD_freeLegacyStreamContext(dctx->legacyContext,
                                   dctx->previousLegacyVersion);
#endif
    if (dctx->ddictSet) {
      ZSTD_freeDDictHashSet(dctx->ddictSet, cMem);
      dctx->ddictSet = NULL;
    }
    ZSTD_customFree(dctx, cMem);
    return 0;
  }
}

/* no longer useful */
DEVICE_INLINE void ZSTD_copyDCtx(ZSTD_DCtx *dstDCtx, const ZSTD_DCtx *srcDCtx) {
  size_t const toCopy = (size_t)((char *)(&dstDCtx->inBuff) - (char *)dstDCtx);
  ZSTD_memcpy(dstDCtx, srcDCtx, toCopy); /* no need to copy workspace */
}

/* Given a dctx with a digested frame params, re-selects the correct ZSTD_DDict
 * based on the requested dict ID from the frame. If there exists a reference to
 * the correct ZSTD_DDict, then accordingly sets the ddict to be used to
 * decompress the frame.
 *
 * If no DDict is found, then no action is taken, and the ZSTD_DCtx::ddict
 * remains as-is.
 *
 * ZSTD_d_refMultipleDDicts must be enabled for this function to be called.
 */
DEVICE_INLINE void ZSTD_DCtx_selectFrameDDict(ZSTD_DCtx *dctx) {
  assert(dctx->refMultipleDDicts && dctx->ddictSet);
  DEBUGLOG(4, "Adjusting DDict based on requested dict ID from frame");
  if (dctx->ddict) {
    const ZSTD_DDict *frameDDict =
        ZSTD_DDictHashSet_getDDict(dctx->ddictSet, dctx->fParams.dictID);
    if (frameDDict) {
      DEBUGLOG(4, "DDict found!");
      ZSTD_clearDict(dctx);
      dctx->dictID = dctx->fParams.dictID;
      dctx->ddict = frameDDict;
      dctx->dictUses = ZSTD_use_indefinitely;
    }
  }
}

/*-*************************************************************
 *   Frame header decoding
 ***************************************************************/

/*! ZSTD_isFrame() :
 *  Tells if the content of `buffer` starts with a valid Frame Identifier.
 *  Note : Frame Identifier is 4 bytes. If `size < 4`, @return will always be 0.
 *  Note 2 : Legacy Frame Identifiers are considered valid only if Legacy
 * Support is enabled. Note 3 : Skippable Frame Identifiers are considered
 * valid. */
DEVICE_INLINE unsigned ZSTD_isFrame(const void *buffer, size_t size) {
  if (size < ZSTD_FRAMEIDSIZE)
    return 0;
  {
    U32 const magic = MEM_readLE32(buffer);
    if (magic == ZSTD_MAGICNUMBER)
      return 1;
    if ((magic & ZSTD_MAGIC_SKIPPABLE_MASK) == ZSTD_MAGIC_SKIPPABLE_START)
      return 1;
  }
#if defined(ZSTD_LEGACY_SUPPORT) && (ZSTD_LEGACY_SUPPORT >= 1)
  if (ZSTD_isLegacy(buffer, size))
    return 1;
#endif
  return 0;
}

/*! ZSTD_isSkippableFrame() :
 *  Tells if the content of `buffer` starts with a valid Frame Identifier for a
 * skippable frame. Note : Frame Identifier is 4 bytes. If `size < 4`, @return
 * will always be 0.
 */
DEVICE_INLINE unsigned ZSTD_isSkippableFrame(const void *buffer, size_t size) {
  if (size < ZSTD_FRAMEIDSIZE)
    return 0;
  {
    U32 const magic = MEM_readLE32(buffer);
    if ((magic & ZSTD_MAGIC_SKIPPABLE_MASK) == ZSTD_MAGIC_SKIPPABLE_START)
      return 1;
  }
  return 0;
}

/** ZSTD_frameHeaderSize_internal() :
 *  srcSize must be large enough to reach header size fields.
 *  note : only works for formats ZSTD_f_zstd1 and ZSTD_f_zstd1_magicless.
 * @return : size of the Frame Header
 *           or an error code, which can be tested with ZSTD_isError() */
DEVICE_INLINE size_t ZSTD_frameHeaderSize_internal(const void *src,
                                                   size_t srcSize,
                                                   ZSTD_format_e format) {
  size_t const minInputSize = ZSTD_startingInputLength(format);
  RETURN_ERROR_IF(srcSize < minInputSize, srcSize_wrong, "");

  {
    BYTE const fhd = ((const BYTE *)src)[minInputSize - 1];
    U32 const dictID = fhd & 3;
    U32 const singleSegment = (fhd >> 5) & 1;
    U32 const fcsId = fhd >> 6;
    return minInputSize + !singleSegment + ZSTD_did_fieldSize[dictID] +
           ZSTD_fcs_fieldSize[fcsId] + (singleSegment && !fcsId);
  }
}

/** ZSTD_frameHeaderSize() :
 *  srcSize must be >= ZSTD_frameHeaderSize_prefix.
 * @return : size of the Frame Header,
 *           or an error code (if srcSize is too small) */
DEVICE_INLINE size_t ZSTD_frameHeaderSize(const void *src, size_t srcSize) {
  return ZSTD_frameHeaderSize_internal(src, srcSize, ZSTD_f_zstd1);
}

constexpr __device__ unsigned long long ZSTD_CONTENTSIZE_UNKNOWN(0ULL - 1);
constexpr __device__ unsigned long long ZSTD_CONTENTSIZE_ERROR(0ULL - 2);

/** ZSTD_getFrameHeader_advanced() :
 *  decode Frame Header, or require larger `srcSize`.
 *  note : only works for formats ZSTD_f_zstd1 and ZSTD_f_zstd1_magicless
 * @return : 0, `zfhPtr` is correctly filled,
 *          >0, `srcSize` is too small, value is wanted `srcSize` amount,
 **           or an error code, which can be tested using ZSTD_isError() */
DEVICE_INLINE size_t ZSTD_getFrameHeader_advanced(ZSTD_FrameHeader *zfhPtr,
                                                  const void *src,
                                                  size_t srcSize,
                                                  ZSTD_format_e format) {
  const BYTE *ip = (const BYTE *)src;
  size_t const minInputSize = ZSTD_startingInputLength(format);

  DEBUGLOG(5, "ZSTD_getFrameHeader_advanced: minInputSize = %zu, srcSize = %zu",
           minInputSize, srcSize);

  if (srcSize > 0) {
    /* note : technically could be considered an assert(), since it's an invalid
     * entry */
    RETURN_ERROR_IF(src == NULL, GENERIC,
                    "invalid parameter : src==NULL, but srcSize>0");
  }
  if (srcSize < minInputSize) {
    if (srcSize > 0 && format != ZSTD_f_zstd1_magicless) {
      /* when receiving less than @minInputSize bytes,
       * control these bytes at least correspond to a supported magic number
       * in order to error out early if they don't.
       **/
      size_t const toCopy = MIN(4, srcSize);
      unsigned char hbuf[4];
      MEM_writeLE32(hbuf, ZSTD_MAGICNUMBER);
      assert(src != NULL);
      ZSTD_memcpy(hbuf, src, toCopy);
      if (MEM_readLE32(hbuf) != ZSTD_MAGICNUMBER) {
        /* not a zstd frame : let's check if it's a skippable frame */
        MEM_writeLE32(hbuf, ZSTD_MAGIC_SKIPPABLE_START);
        ZSTD_memcpy(hbuf, src, toCopy);
        if ((MEM_readLE32(hbuf) & ZSTD_MAGIC_SKIPPABLE_MASK) !=
            ZSTD_MAGIC_SKIPPABLE_START) {
          RETURN_ERROR(
              prefix_unknown,
              "first bytes don't correspond to any supported magic number");
        }
      }
    }
    return minInputSize;
  }

  ZSTD_memset(
      zfhPtr, 0,
      sizeof(*zfhPtr)); /* not strictly necessary, but static analyzers may not
                           understand that zfhPtr will be read only if return
                           value is zero, since they are 2 different signals */
  if ((format != ZSTD_f_zstd1_magicless) &&
      (MEM_readLE32(src) != ZSTD_MAGICNUMBER)) {
    if ((MEM_readLE32(src) & ZSTD_MAGIC_SKIPPABLE_MASK) ==
        ZSTD_MAGIC_SKIPPABLE_START) {
      /* skippable frame */
      if (srcSize < ZSTD_SKIPPABLEHEADERSIZE)
        return ZSTD_SKIPPABLEHEADERSIZE; /* magic number + frame length */
      ZSTD_memset(zfhPtr, 0, sizeof(*zfhPtr));
      zfhPtr->frameType = ZSTD_skippableFrame;
      zfhPtr->dictID = MEM_readLE32(src) - ZSTD_MAGIC_SKIPPABLE_START;
      zfhPtr->headerSize = ZSTD_SKIPPABLEHEADERSIZE;
      zfhPtr->frameContentSize =
          MEM_readLE32((const char *)src + ZSTD_FRAMEIDSIZE);
      return 0;
    }
    RETURN_ERROR(prefix_unknown, "");
  }

  /* ensure there is enough `srcSize` to fully read/decode frame header */
  {
    size_t const fhsize = ZSTD_frameHeaderSize_internal(src, srcSize, format);
    if (srcSize < fhsize)
      return fhsize;
    zfhPtr->headerSize = (U32)fhsize;
  }

  {
    BYTE const fhdByte = ip[minInputSize - 1];
    size_t pos = minInputSize;
    U32 const dictIDSizeCode = fhdByte & 3;
    U32 const checksumFlag = (fhdByte >> 2) & 1;
    U32 const singleSegment = (fhdByte >> 5) & 1;
    U32 const fcsID = fhdByte >> 6;
    U64 windowSize = 0;
    U32 dictID = 0;
    U64 frameContentSize = ZSTD_CONTENTSIZE_UNKNOWN;
    RETURN_ERROR_IF((fhdByte & 0x08) != 0, frameParameter_unsupported,
                    "reserved bits, must be zero");

    if (!singleSegment) {
      BYTE const wlByte = ip[pos++];
      U32 const windowLog = (wlByte >> 3) + ZSTD_WINDOWLOG_ABSOLUTEMIN;
      RETURN_ERROR_IF(windowLog > ZSTD_WINDOWLOG_MAX,
                      frameParameter_windowTooLarge, "");
      windowSize = (1ULL << windowLog);
      windowSize += (windowSize >> 3) * (wlByte & 7);
    }
    switch (dictIDSizeCode) {
    default:
      assert(0); /* impossible */
      ZSTD_FALLTHROUGH;
    case 0:
      break;
    case 1:
      dictID = ip[pos];
      pos++;
      break;
    case 2:
      dictID = MEM_readLE16(ip + pos);
      pos += 2;
      break;
    case 3:
      dictID = MEM_readLE32(ip + pos);
      pos += 4;
      break;
    }
    switch (fcsID) {
    default:
      assert(0); /* impossible */
      ZSTD_FALLTHROUGH;
    case 0:
      if (singleSegment)
        frameContentSize = ip[pos];
      break;
    case 1:
      frameContentSize = MEM_readLE16(ip + pos) + 256;
      break;
    case 2:
      frameContentSize = MEM_readLE32(ip + pos);
      break;
    case 3:
      frameContentSize = MEM_readLE64(ip + pos);
      break;
    }
    if (singleSegment)
      windowSize = frameContentSize;

    zfhPtr->frameType = ZSTD_frame;
    zfhPtr->frameContentSize = frameContentSize;
    zfhPtr->windowSize = windowSize;
    zfhPtr->blockSizeMax = (unsigned)MIN(windowSize, ZSTD_BLOCKSIZE_MAX);
    zfhPtr->dictID = dictID;
    zfhPtr->checksumFlag = checksumFlag;
  }
  return 0;
}

/** ZSTD_getFrameHeader() :
 *  decode Frame Header, or require larger `srcSize`.
 *  note : this function does not consume input, it only reads it.
 * @return : 0, `zfhPtr` is correctly filled,
 *          >0, `srcSize` is too small, value is wanted `srcSize` amount,
 *           or an error code, which can be tested using ZSTD_isError() */
DEVICE_INLINE size_t ZSTD_getFrameHeader(ZSTD_FrameHeader *zfhPtr,
                                         const void *src, size_t srcSize) {
  return ZSTD_getFrameHeader_advanced(zfhPtr, src, srcSize, ZSTD_f_zstd1);
}

/** ZSTD_getFrameContentSize() :
 *  compatible with legacy mode
 * @return : decompressed size of the single frame pointed to be `src` if known,
 * otherwise
 *         - ZSTD_CONTENTSIZE_UNKNOWN if the size cannot be determined
 *         - ZSTD_CONTENTSIZE_ERROR if an error occurred (e.g. invalid magic
 * number, srcSize too small) */
DEVICE_INLINE unsigned long long ZSTD_getFrameContentSize(const void *src,
                                                          size_t srcSize) {
#if defined(ZSTD_LEGACY_SUPPORT) && (ZSTD_LEGACY_SUPPORT >= 1)
  if (ZSTD_isLegacy(src, srcSize)) {
    unsigned long long const ret =
        ZSTD_getDecompressedSize_legacy(src, srcSize);
    return ret == 0 ? ZSTD_CONTENTSIZE_UNKNOWN : ret;
  }
#endif
  {
    ZSTD_FrameHeader zfh;
    if (ZSTD_getFrameHeader(&zfh, src, srcSize) != 0)
      return ZSTD_CONTENTSIZE_ERROR;
    if (zfh.frameType == ZSTD_skippableFrame) {
      return 0;
    } else {
      return zfh.frameContentSize;
    }
  }
}

DEVICE_INLINE size_t readSkippableFrameSize(void const *src, size_t srcSize) {
  size_t const skippableHeaderSize = ZSTD_SKIPPABLEHEADERSIZE;
  U32 sizeU32;

  RETURN_ERROR_IF(srcSize < ZSTD_SKIPPABLEHEADERSIZE, srcSize_wrong, "");

  sizeU32 = MEM_readLE32((BYTE const *)src + ZSTD_FRAMEIDSIZE);
  RETURN_ERROR_IF((U32)(sizeU32 + ZSTD_SKIPPABLEHEADERSIZE) < sizeU32,
                  frameParameter_unsupported, "");
  {
    size_t const skippableSize = skippableHeaderSize + sizeU32;
    RETURN_ERROR_IF(skippableSize > srcSize, srcSize_wrong, "");
    return skippableSize;
  }
}

/*! ZSTD_readSkippableFrame() :
 * Retrieves content of a skippable frame, and writes it to dst buffer.
 *
 * The parameter magicVariant will receive the magicVariant that was supplied
 * when the frame was written, i.e. magicNumber - ZSTD_MAGIC_SKIPPABLE_START.
 * This can be NULL if the caller is not interested in the magicVariant.
 *
 * Returns an error if destination buffer is not large enough, or if this is not
 * a valid skippable frame.
 *
 * @return : number of bytes written or a ZSTD error.
 */
DEVICE_INLINE size_t
ZSTD_readSkippableFrame(void *dst, size_t dstCapacity,
                        unsigned *magicVariant, /* optional, can be NULL */
                        const void *src, size_t srcSize) {
  RETURN_ERROR_IF(srcSize < ZSTD_SKIPPABLEHEADERSIZE, srcSize_wrong, "");

  {
    U32 const magicNumber = MEM_readLE32(src);
    size_t skippableFrameSize = readSkippableFrameSize(src, srcSize);
    size_t skippableContentSize = skippableFrameSize - ZSTD_SKIPPABLEHEADERSIZE;

    /* check input validity */
    RETURN_ERROR_IF(!ZSTD_isSkippableFrame(src, srcSize),
                    frameParameter_unsupported, "");
    RETURN_ERROR_IF(skippableFrameSize < ZSTD_SKIPPABLEHEADERSIZE ||
                        skippableFrameSize > srcSize,
                    srcSize_wrong, "");
    RETURN_ERROR_IF(skippableContentSize > dstCapacity, dstSize_tooSmall, "");

    /* deliver payload */
    if (skippableContentSize > 0 && dst != NULL)
      ZSTD_memcpy(dst, (const BYTE *)src + ZSTD_SKIPPABLEHEADERSIZE,
                  skippableContentSize);
    if (magicVariant != NULL)
      *magicVariant = magicNumber - ZSTD_MAGIC_SKIPPABLE_START;
    return skippableContentSize;
  }
}

/** ZSTD_decodeFrameHeader() :
 * `headerSize` must be the size provided by ZSTD_frameHeaderSize().
 * If multiple DDict references are enabled, also will choose the correct DDict
 * to use.
 * @return : 0 if success, or an error code, which can be tested using
 * ZSTD_isError() */
DEVICE_INLINE size_t ZSTD_decodeFrameHeader(ZSTD_DCtx *dctx, const void *src,
                                            size_t headerSize) {
  size_t const result = ZSTD_getFrameHeader_advanced(&(dctx->fParams), src,
                                                     headerSize, dctx->format);
  if (ZSTD_isError(result))
    return result; /* invalid header */
  RETURN_ERROR_IF(result > 0, srcSize_wrong, "headerSize too small");

  /* Reference DDict requested by frame if dctx references multiple ddicts */
  if (dctx->refMultipleDDicts == ZSTD_rmd_refMultipleDDicts && dctx->ddictSet) {
    ZSTD_DCtx_selectFrameDDict(dctx);
  }

#ifndef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  /* Skip the dictID check in fuzzing mode, because it makes the search
   * harder.
   */
  RETURN_ERROR_IF(dctx->fParams.dictID &&
                      (dctx->dictID != dctx->fParams.dictID),
                  dictionary_wrong, "");
#endif
  dctx->validateChecksum =
      (dctx->fParams.checksumFlag && !dctx->forceIgnoreChecksum) ? 1 : 0;
  if (dctx->validateChecksum)
    XXH64_reset(&dctx->xxhState, 0);
  dctx->processedCSize += headerSize;
  return 0;
}

DEVICE_INLINE ZSTD_frameSizeInfo ZSTD_errorFrameSizeInfo(size_t ret) {
  ZSTD_frameSizeInfo frameSizeInfo;
  frameSizeInfo.compressedSize = ret;
  frameSizeInfo.decompressedBound = ZSTD_CONTENTSIZE_ERROR;
  return frameSizeInfo;
}

DEVICE_INLINE ZSTD_frameSizeInfo ZSTD_findFrameSizeInfo(const void *src,
                                                        size_t srcSize,
                                                        ZSTD_format_e format) {
  ZSTD_frameSizeInfo frameSizeInfo;
  ZSTD_memset(&frameSizeInfo, 0, sizeof(ZSTD_frameSizeInfo));

#if defined(ZSTD_LEGACY_SUPPORT) && (ZSTD_LEGACY_SUPPORT >= 1)
  if (format == ZSTD_f_zstd1 && ZSTD_isLegacy(src, srcSize))
    return ZSTD_findFrameSizeInfoLegacy(src, srcSize);
#endif

  if (format == ZSTD_f_zstd1 && (srcSize >= ZSTD_SKIPPABLEHEADERSIZE) &&
      (MEM_readLE32(src) & ZSTD_MAGIC_SKIPPABLE_MASK) ==
          ZSTD_MAGIC_SKIPPABLE_START) {
    frameSizeInfo.compressedSize = readSkippableFrameSize(src, srcSize);
    assert(ZSTD_isError(frameSizeInfo.compressedSize) ||
           frameSizeInfo.compressedSize <= srcSize);
    return frameSizeInfo;
  } else {
    const BYTE *ip = (const BYTE *)src;
    const BYTE *const ipstart = ip;
    size_t remainingSize = srcSize;
    size_t nbBlocks = 0;
    ZSTD_FrameHeader zfh;

    /* Extract Frame Header */
    {
      size_t const ret =
          ZSTD_getFrameHeader_advanced(&zfh, src, srcSize, format);
      if (ZSTD_isError(ret))
        return ZSTD_errorFrameSizeInfo(ret);
      if (ret > 0)
        return ZSTD_errorFrameSizeInfo(ERROR(srcSize_wrong));
    }

    ip += zfh.headerSize;
    remainingSize -= zfh.headerSize;

    /* Iterate over each block */
    while (1) {
      blockProperties_t blockProperties;
      size_t const cBlockSize =
          ZSTD_getcBlockSize(ip, remainingSize, &blockProperties);
      if (ZSTD_isError(cBlockSize))
        return ZSTD_errorFrameSizeInfo(cBlockSize);

      if (ZSTD_blockHeaderSize + cBlockSize > remainingSize)
        return ZSTD_errorFrameSizeInfo(ERROR(srcSize_wrong));

      ip += ZSTD_blockHeaderSize + cBlockSize;
      remainingSize -= ZSTD_blockHeaderSize + cBlockSize;
      nbBlocks++;

      if (blockProperties.lastBlock)
        break;
    }

    /* Final frame content checksum */
    if (zfh.checksumFlag) {
      if (remainingSize < 4)
        return ZSTD_errorFrameSizeInfo(ERROR(srcSize_wrong));
      ip += 4;
    }

    frameSizeInfo.nbBlocks = nbBlocks;
    frameSizeInfo.compressedSize = (size_t)(ip - ipstart);
    frameSizeInfo.decompressedBound =
        (zfh.frameContentSize != ZSTD_CONTENTSIZE_UNKNOWN)
            ? zfh.frameContentSize
            : (unsigned long long)nbBlocks * zfh.blockSizeMax;
    return frameSizeInfo;
  }
}

DEVICE_INLINE size_t ZSTD_findFrameCompressedSize_advanced(
    const void *src, size_t srcSize, ZSTD_format_e format) {
  ZSTD_frameSizeInfo const frameSizeInfo =
      ZSTD_findFrameSizeInfo(src, srcSize, format);
  return frameSizeInfo.compressedSize;
}

/** ZSTD_findFrameCompressedSize() :
 * See docs in zstd.h
 * Note: compatible with legacy mode */
DEVICE_INLINE size_t ZSTD_findFrameCompressedSize(const void *src,
                                                  size_t srcSize) {
  return ZSTD_findFrameCompressedSize_advanced(src, srcSize, ZSTD_f_zstd1);
}

/** ZSTD_findDecompressedSize() :
 *  `srcSize` must be the exact length of some number of ZSTD compressed and/or
 *      skippable frames
 *  note: compatible with legacy mode
 * @return : decompressed size of the frames contained */
DEVICE_INLINE unsigned long long ZSTD_findDecompressedSize(const void *src,
                                                           size_t srcSize) {
  unsigned long long totalDstSize = 0;

  while (srcSize >= ZSTD_startingInputLength(ZSTD_f_zstd1)) {
    U32 const magicNumber = MEM_readLE32(src);

    if ((magicNumber & ZSTD_MAGIC_SKIPPABLE_MASK) ==
        ZSTD_MAGIC_SKIPPABLE_START) {
      size_t const skippableSize = readSkippableFrameSize(src, srcSize);
      if (ZSTD_isError(skippableSize))
        return ZSTD_CONTENTSIZE_ERROR;
      assert(skippableSize <= srcSize);

      src = (const BYTE *)src + skippableSize;
      srcSize -= skippableSize;
      continue;
    }

    {
      unsigned long long const fcs = ZSTD_getFrameContentSize(src, srcSize);
      if (fcs >= ZSTD_CONTENTSIZE_ERROR)
        return fcs;

      if (totalDstSize + fcs < totalDstSize)
        return ZSTD_CONTENTSIZE_ERROR; /* check for overflow */
      totalDstSize += fcs;
    }
    /* skip to next frame */
    {
      size_t const frameSrcSize = ZSTD_findFrameCompressedSize(src, srcSize);
      if (ZSTD_isError(frameSrcSize))
        return ZSTD_CONTENTSIZE_ERROR;
      assert(frameSrcSize <= srcSize);

      src = (const BYTE *)src + frameSrcSize;
      srcSize -= frameSrcSize;
    }
  } /* while (srcSize >= ZSTD_frameHeaderSize_prefix) */

  if (srcSize)
    return ZSTD_CONTENTSIZE_ERROR;

  return totalDstSize;
}

/** ZSTD_getDecompressedSize() :
 *  compatible with legacy mode
 * @return : decompressed size if known, 0 otherwise
             note : 0 can mean any of the following :
                   - frame content is empty
                   - decompressed size field is not present in frame header
                   - frame header unknown / not supported
                   - frame header not complete (`srcSize` too small) */
DEVICE_INLINE unsigned long long ZSTD_getDecompressedSize(const void *src,
                                                          size_t srcSize) {
  unsigned long long const ret = ZSTD_getFrameContentSize(src, srcSize);
  ZSTD_STATIC_ASSERT(ZSTD_CONTENTSIZE_ERROR < ZSTD_CONTENTSIZE_UNKNOWN);
  return (ret >= ZSTD_CONTENTSIZE_ERROR) ? 0 : ret;
}

/** ZSTD_decompressBound() :
 *  compatible with legacy mode
 *  `src` must point to the start of a ZSTD frame or a skippable frame
 *  `srcSize` must be at least as large as the frame contained
 *  @return : the maximum decompressed size of the compressed source
 */
DEVICE_INLINE unsigned long long ZSTD_decompressBound(const void *src,
                                                      size_t srcSize) {
  unsigned long long bound = 0;
  /* Iterate over each frame */
  while (srcSize > 0) {
    ZSTD_frameSizeInfo const frameSizeInfo =
        ZSTD_findFrameSizeInfo(src, srcSize, ZSTD_f_zstd1);
    size_t const compressedSize = frameSizeInfo.compressedSize;
    unsigned long long const decompressedBound =
        frameSizeInfo.decompressedBound;
    if (ZSTD_isError(compressedSize) ||
        decompressedBound == ZSTD_CONTENTSIZE_ERROR)
      return ZSTD_CONTENTSIZE_ERROR;
    assert(srcSize >= compressedSize);
    src = (const BYTE *)src + compressedSize;
    srcSize -= compressedSize;
    bound += decompressedBound;
  }
  return bound;
}

DEVICE_INLINE size_t ZSTD_decompressionMargin(void const *src, size_t srcSize) {
  size_t margin = 0;
  unsigned maxBlockSize = 0;

  /* Iterate over each frame */
  while (srcSize > 0) {
    ZSTD_frameSizeInfo const frameSizeInfo =
        ZSTD_findFrameSizeInfo(src, srcSize, ZSTD_f_zstd1);
    size_t const compressedSize = frameSizeInfo.compressedSize;
    unsigned long long const decompressedBound =
        frameSizeInfo.decompressedBound;
    ZSTD_FrameHeader zfh;

    FORWARD_IF_ERROR(ZSTD_getFrameHeader(&zfh, src, srcSize), "");
    if (ZSTD_isError(compressedSize) ||
        decompressedBound == ZSTD_CONTENTSIZE_ERROR)
      return ERROR(corruption_detected);

    if (zfh.frameType == ZSTD_frame) {
      /* Add the frame header to our margin */
      margin += zfh.headerSize;
      /* Add the checksum to our margin */
      margin += zfh.checksumFlag ? 4 : 0;
      /* Add 3 bytes per block */
      margin += 3 * frameSizeInfo.nbBlocks;

      /* Compute the max block size */
      maxBlockSize = MAX(maxBlockSize, zfh.blockSizeMax);
    } else {
      assert(zfh.frameType == ZSTD_skippableFrame);
      /* Add the entire skippable frame size to our margin. */
      margin += compressedSize;
    }

    assert(srcSize >= compressedSize);
    src = (const BYTE *)src + compressedSize;
    srcSize -= compressedSize;
  }

  /* Add the max block size back to the margin. */
  margin += maxBlockSize;

  return margin;
}

/*-*************************************************************
 *   Frame decoding
 ***************************************************************/

/** ZSTD_insertBlock() :
 *  insert `src` block into `dctx` history. Useful to track uncompressed blocks.
 */
DEVICE_INLINE size_t ZSTD_insertBlock(ZSTD_DCtx *dctx, const void *blockStart,
                                      size_t blockSize) {
  DEBUGLOG(5, "ZSTD_insertBlock: %u bytes", (unsigned)blockSize);
  ZSTD_checkContinuity(dctx, blockStart, blockSize);
  dctx->previousDstEnd = (const char *)blockStart + blockSize;
  return blockSize;
}

DEVICE_INLINE size_t ZSTD_copyRawBlock(void *dst, size_t dstCapacity,
                                       const void *src, size_t srcSize) {
  DEBUGLOG(5, "ZSTD_copyRawBlock");
  RETURN_ERROR_IF(srcSize > dstCapacity, dstSize_tooSmall, "");
  if (dst == NULL) {
    if (srcSize == 0)
      return 0;
    RETURN_ERROR(dstBuffer_null, "");
  }
  ZSTD_memmove(dst, src, srcSize);
  return srcSize;
}

DEVICE_INLINE size_t ZSTD_setRleBlock(void *dst, size_t dstCapacity, BYTE b,
                                      size_t regenSize) {
  RETURN_ERROR_IF(regenSize > dstCapacity, dstSize_tooSmall, "");
  if (dst == NULL) {
    if (regenSize == 0)
      return 0;
    RETURN_ERROR(dstBuffer_null, "");
  }
  ZSTD_memset(dst, b, regenSize);
  return regenSize;
}

DEVICE_INLINE void ZSTD_DCtx_trace_end(ZSTD_DCtx const *dctx,
                                       U64 uncompressedSize, U64 compressedSize,
                                       int streaming) {
  (void)dctx;
  (void)uncompressedSize;
  (void)compressedSize;
  (void)streaming;
}

/* ======   ZSTD_DDict   ====== */

DEVICE_INLINE size_t ZSTD_refDictContent(ZSTD_DCtx *dctx, const void *dict,
                                         size_t dictSize) {
  dctx->dictEnd = dctx->previousDstEnd;
  dctx->virtualStart =
      (const char *)dict - ((const char *)(dctx->previousDstEnd) -
                            (const char *)(dctx->prefixStart));
  dctx->prefixStart = dict;
  dctx->previousDstEnd = (const char *)dict + dictSize;
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  dctx->dictContentBeginForFuzzing = dctx->prefixStart;
  dctx->dictContentEndForFuzzing = dctx->previousDstEnd;
#endif
  return 0;
}

DEVICE_INLINE size_t ZSTD_decompress_insertDictionary(ZSTD_DCtx *dctx,
                                                      const void *dict,
                                                      size_t dictSize) {
  if (dictSize < 8)
    return ZSTD_refDictContent(dctx, dict, dictSize);
  {
    U32 const magic = MEM_readLE32(dict);
    if (magic != ZSTD_MAGIC_DICTIONARY) {
      return ZSTD_refDictContent(dctx, dict, dictSize); /* pure content mode */
    }
  }
  dctx->dictID = MEM_readLE32((const char *)dict + ZSTD_FRAMEIDSIZE);

  /* load entropy tables */
  {
    size_t const eSize = ZSTD_loadDEntropy(&dctx->entropy, dict, dictSize);
    RETURN_ERROR_IF(ZSTD_isError(eSize), dictionary_corrupted, "");
    dict = (const char *)dict + eSize;
    dictSize -= eSize;
  }
  dctx->litEntropy = dctx->fseEntropy = 1;

  /* reference dictionary content */
  return ZSTD_refDictContent(dctx, dict, dictSize);
}

DEVICE_INLINE size_t ZSTD_decompressBegin(ZSTD_DCtx *dctx) {
  assert(dctx != NULL);
  dctx->expected = ZSTD_startingInputLength(
      dctx->format); /* dctx->format must be properly set */
  dctx->stage = ZSTDds_getFrameHeaderSize;
  dctx->processedCSize = 0;
  dctx->decodedSize = 0;
  dctx->previousDstEnd = NULL;
  dctx->prefixStart = NULL;
  dctx->virtualStart = NULL;
  dctx->dictEnd = NULL;
  dctx->entropy.hufTable[0] =
      (HUF_DTable)((ZSTD_HUFFDTABLE_CAPACITY_LOG) *
                   0x1000001); /* cover both little and big endian */
  dctx->litEntropy = dctx->fseEntropy = 0;
  dctx->dictID = 0;
  dctx->bType = bt_reserved;
  dctx->isFrameDecompression = 1;
  ZSTD_STATIC_ASSERT(sizeof(dctx->entropy.rep) == sizeof(repStartValue));
  ZSTD_memcpy(dctx->entropy.rep, repStartValue,
              sizeof(repStartValue)); /* initial repcodes */
  dctx->LLTptr = dctx->entropy.LLTable;
  dctx->MLTptr = dctx->entropy.MLTable;
  dctx->OFTptr = dctx->entropy.OFTable;
  dctx->HUFptr = dctx->entropy.hufTable;
  return 0;
}

DEVICE_INLINE size_t ZSTD_decompressBegin_usingDict(ZSTD_DCtx *dctx,
                                                    const void *dict,
                                                    size_t dictSize) {
  FORWARD_IF_ERROR(ZSTD_decompressBegin(dctx), "");
  if (dict && dictSize)
    RETURN_ERROR_IF(
        ZSTD_isError(ZSTD_decompress_insertDictionary(dctx, dict, dictSize)),
        dictionary_corrupted, "");
  return 0;
}

DEVICE_INLINE size_t ZSTD_decompressBegin_usingDDict(ZSTD_DCtx *dctx,
                                                     const ZSTD_DDict *ddict) {
  DEBUGLOG(4, "ZSTD_decompressBegin_usingDDict");
  assert(dctx != NULL);
  if (ddict) {
    const char *const dictStart = (const char *)ZSTD_DDict_dictContent(ddict);
    size_t const dictSize = ZSTD_DDict_dictSize(ddict);
    const void *const dictEnd = dictStart + dictSize;
    dctx->ddictIsCold = (dctx->dictEnd != dictEnd);
    DEBUGLOG(4, "DDict is %s", dctx->ddictIsCold ? "~cold~" : "hot!");
  }
  FORWARD_IF_ERROR(ZSTD_decompressBegin(dctx), "");
  if (ddict) { /* NULL ddict is equivalent to no dictionary */
    ZSTD_copyDDictParameters(dctx, ddict);
  }
  return 0;
}

/*! ZSTD_decompressFrame() :
 * @dctx must be properly initialized
 *  will update *srcPtr and *srcSizePtr,
 *  to make *srcPtr progress by one frame. */
DEVICE_INLINE size_t ZSTD_decompressFrame(ZSTD_DCtx *dctx, void *dst,
                                          size_t dstCapacity,
                                          const void **srcPtr,
                                          size_t *srcSizePtr) {
  const BYTE *const istart = (const BYTE *)(*srcPtr);
  const BYTE *ip = istart;
  BYTE *const ostart = (BYTE *)dst;
  BYTE *const oend = dstCapacity != 0 ? ostart + dstCapacity : ostart;
  BYTE *op = ostart;
  size_t remainingSrcSize = *srcSizePtr;

  DEBUGLOG(4, "ZSTD_decompressFrame (srcSize:%i)", (int)*srcSizePtr);

  /* check */
  RETURN_ERROR_IF(remainingSrcSize < ZSTD_FRAMEHEADERSIZE_MIN(dctx->format) +
                                         ZSTD_blockHeaderSize,
                  srcSize_wrong, "");

  /* Frame Header */
  {
    size_t const frameHeaderSize = ZSTD_frameHeaderSize_internal(
        ip, ZSTD_FRAMEHEADERSIZE_PREFIX(dctx->format), dctx->format);
    if (ZSTD_isError(frameHeaderSize))
      return frameHeaderSize;
    RETURN_ERROR_IF(remainingSrcSize < frameHeaderSize + ZSTD_blockHeaderSize,
                    srcSize_wrong, "");
    FORWARD_IF_ERROR(ZSTD_decodeFrameHeader(dctx, ip, frameHeaderSize), "");
    ip += frameHeaderSize;
    remainingSrcSize -= frameHeaderSize;
  }

  /* Shrink the blockSizeMax if enabled */
  if (dctx->maxBlockSizeParam != 0)
    dctx->fParams.blockSizeMax =
        MIN(dctx->fParams.blockSizeMax, (unsigned)dctx->maxBlockSizeParam);

  /* Loop on each block */
  while (1) {
    BYTE *oBlockEnd = oend;
    size_t decodedSize;
    blockProperties_t blockProperties;
    size_t const cBlockSize =
        ZSTD_getcBlockSize(ip, remainingSrcSize, &blockProperties);
    if (ZSTD_isError(cBlockSize))
      return cBlockSize;

    ip += ZSTD_blockHeaderSize;
    remainingSrcSize -= ZSTD_blockHeaderSize;
    RETURN_ERROR_IF(cBlockSize > remainingSrcSize, srcSize_wrong, "");

    if (ip >= op && ip < oBlockEnd) {
      /* We are decompressing in-place. Limit the output pointer so that we
       * don't overwrite the block that we are currently reading. This will
       * fail decompression if the input & output pointers aren't spaced
       * far enough apart.
       *
       * This is important to set, even when the pointers are far enough
       * apart, because ZSTD_decompressBlock_internal() can decide to store
       * literals in the output buffer, after the block it is decompressing.
       * Since we don't want anything to overwrite our input, we have to tell
       * ZSTD_decompressBlock_internal to never write past ip.
       *
       * See ZSTD_allocateLiteralsBuffer() for reference.
       */
      oBlockEnd = op + (ip - op);
    }

    switch (blockProperties.blockType) {
    case bt_compressed:
      assert(dctx->isFrameDecompression == 1);
      decodedSize = ZSTD_decompressBlock_internal(
          dctx, op, (size_t)(oBlockEnd - op), ip, cBlockSize, not_streaming);
      break;
    case bt_raw:
      /* Use oend instead of oBlockEnd because this function is safe to overlap.
       * It uses memmove. */
      decodedSize = ZSTD_copyRawBlock(op, (size_t)(oend - op), ip, cBlockSize);
      break;
    case bt_rle:
      decodedSize = ZSTD_setRleBlock(op, (size_t)(oBlockEnd - op), *ip,
                                     blockProperties.origSize);
      break;
    case bt_reserved:
    default:
      RETURN_ERROR(corruption_detected, "invalid block type");
    }
    FORWARD_IF_ERROR(decodedSize, "Block decompression failure");
    DEBUGLOG(5, "Decompressed block of dSize = %u", (unsigned)decodedSize);
    if (dctx->validateChecksum) {
      XXH64_update(&dctx->xxhState, op, decodedSize);
    }
    if (decodedSize) /* support dst = NULL,0 */ {
      op += decodedSize;
    }
    assert(ip != NULL);
    ip += cBlockSize;
    remainingSrcSize -= cBlockSize;
    if (blockProperties.lastBlock)
      break;
  }

  if (dctx->fParams.frameContentSize != ZSTD_CONTENTSIZE_UNKNOWN) {
    RETURN_ERROR_IF((U64)(op - ostart) != dctx->fParams.frameContentSize,
                    corruption_detected, "");
  }
  if (dctx->fParams.checksumFlag) { /* Frame content checksum verification */
    RETURN_ERROR_IF(remainingSrcSize < 4, checksum_wrong, "");
    if (!dctx->forceIgnoreChecksum) {
      U32 const checkCalc = (U32)XXH64_digest(&dctx->xxhState);
      U32 checkRead;
      checkRead = MEM_readLE32(ip);
      RETURN_ERROR_IF(checkRead != checkCalc, checksum_wrong, "");
    }
    ip += 4;
    remainingSrcSize -= 4;
  }
  ZSTD_DCtx_trace_end(dctx, (U64)(op - ostart), (U64)(ip - istart),
                      /* streaming */ 0);
  /* Allow caller to get size read */
  DEBUGLOG(4,
           "ZSTD_decompressFrame: decompressed frame of size %i, consuming %i "
           "bytes of input",
           (int)(op - ostart), (int)(ip - (const BYTE *)*srcPtr));
  *srcPtr = ip;
  *srcSizePtr = remainingSrcSize;
  return (size_t)(op - ostart);
}

DEVICE_INLINE
ZSTD_ALLOW_POINTER_OVERFLOW_ATTR
size_t ZSTD_decompressMultiFrame(ZSTD_DCtx *dctx, void *dst, size_t dstCapacity,
                                 const void *src, size_t srcSize,
                                 const void *dict, size_t dictSize,
                                 const ZSTD_DDict *ddict) {
  void *const dststart = dst;
  int moreThan1Frame = 0;

  DEBUGLOG(5, "ZSTD_decompressMultiFrame");
  assert(dict == NULL ||
         ddict == NULL); /* either dict or ddict set, not both */

  if (ddict) {
    dict = ZSTD_DDict_dictContent(ddict);
    dictSize = ZSTD_DDict_dictSize(ddict);
  }

  while (srcSize >= ZSTD_startingInputLength(dctx->format)) {

#if defined(ZSTD_LEGACY_SUPPORT) && (ZSTD_LEGACY_SUPPORT >= 1)
    if (dctx->format == ZSTD_f_zstd1 && ZSTD_isLegacy(src, srcSize)) {
      size_t decodedSize;
      size_t const frameSize = ZSTD_findFrameCompressedSizeLegacy(src, srcSize);
      if (ZSTD_isError(frameSize))
        return frameSize;
      RETURN_ERROR_IF(dctx->staticSize, memory_allocation,
                      "legacy support is not compatible with static dctx");

      decodedSize = ZSTD_decompressLegacy(dst, dstCapacity, src, frameSize,
                                          dict, dictSize);
      if (ZSTD_isError(decodedSize))
        return decodedSize;

      {
        unsigned long long const expectedSize =
            ZSTD_getFrameContentSize(src, srcSize);
        RETURN_ERROR_IF(expectedSize == ZSTD_CONTENTSIZE_ERROR,
                        corruption_detected, "Corrupted frame header!");
        if (expectedSize != ZSTD_CONTENTSIZE_UNKNOWN) {
          RETURN_ERROR_IF(expectedSize != decodedSize, corruption_detected,
                          "Frame header size does not match decoded size!");
        }
      }

      assert(decodedSize <= dstCapacity);
      dst = (BYTE *)dst + decodedSize;
      dstCapacity -= decodedSize;

      src = (const BYTE *)src + frameSize;
      srcSize -= frameSize;

      continue;
    }
#endif

    if (dctx->format == ZSTD_f_zstd1 && srcSize >= 4) {
      U32 const magicNumber = MEM_readLE32(src);
      DEBUGLOG(5, "reading magic number %08X", (unsigned)magicNumber);
      if ((magicNumber & ZSTD_MAGIC_SKIPPABLE_MASK) ==
          ZSTD_MAGIC_SKIPPABLE_START) {
        /* skippable frame detected : skip it */
        size_t const skippableSize = readSkippableFrameSize(src, srcSize);
        FORWARD_IF_ERROR(skippableSize, "invalid skippable frame");
        assert(skippableSize <= srcSize);

        src = (const BYTE *)src + skippableSize;
        srcSize -= skippableSize;
        continue; /* check next frame */
      }
    }

    if (ddict) {
      /* we were called from ZSTD_decompress_usingDDict */
      FORWARD_IF_ERROR(ZSTD_decompressBegin_usingDDict(dctx, ddict), "");
    } else {
      /* this will initialize correctly with no dict if dict == NULL, so
       * use this in all cases but ddict */
      FORWARD_IF_ERROR(ZSTD_decompressBegin_usingDict(dctx, dict, dictSize),
                       "");
    }
    ZSTD_checkContinuity(dctx, dst, dstCapacity);

    {
      const size_t res =
          ZSTD_decompressFrame(dctx, dst, dstCapacity, &src, &srcSize);
      RETURN_ERROR_IF(
          (ZSTD_getErrorCode(res) == ZSTD_error_prefix_unknown) &&
              (moreThan1Frame == 1),
          srcSize_wrong,
          "At least one frame successfully completed, "
          "but following bytes are garbage: "
          "it's more likely to be a srcSize error, "
          "specifying more input bytes than size of frame(s). "
          "Note: one could be unlucky, it might be a corruption error instead, "
          "happening right at the place where we expect zstd magic bytes. "
          "But this is _much_ less likely than a srcSize field error.");
      if (ZSTD_isError(res))
        return res;
      assert(res <= dstCapacity);
      if (res != 0)
        dst = (BYTE *)dst + res;
      dstCapacity -= res;
    }
    moreThan1Frame = 1;
  } /* while (srcSize >= ZSTD_frameHeaderSize_prefix) */

  RETURN_ERROR_IF(srcSize, srcSize_wrong, "input not entirely consumed");

  return (size_t)((BYTE *)dst - (BYTE *)dststart);
}

DEVICE_INLINE size_t ZSTD_decompress_usingDict(ZSTD_DCtx *dctx, void *dst,
                                               size_t dstCapacity,
                                               const void *src, size_t srcSize,
                                               const void *dict,
                                               size_t dictSize) {
  return ZSTD_decompressMultiFrame(dctx, dst, dstCapacity, src, srcSize, dict,
                                   dictSize, NULL);
}

DEVICE_INLINE ZSTD_DDict const *ZSTD_getDDict(ZSTD_DCtx *dctx) {
  switch (dctx->dictUses) {
  default:
    assert(0 /* Impossible */);
    ZSTD_FALLTHROUGH;
  case ZSTD_dont_use:
    ZSTD_clearDict(dctx);
    return NULL;
  case ZSTD_use_indefinitely:
    return dctx->ddict;
  case ZSTD_use_once:
    dctx->dictUses = ZSTD_dont_use;
    return dctx->ddict;
  }
}

/*! ZSTD_decompress_usingDDict() :
 *   Decompression using a pre-digested Dictionary
 *   Use dictionary without significant overhead. */
DEVICE_INLINE size_t ZSTD_decompress_usingDDict(ZSTD_DCtx *dctx, void *dst,
                                                size_t dstCapacity,
                                                const void *src, size_t srcSize,
                                                const ZSTD_DDict *ddict) {
  /* pass content and size in case legacy frames are encountered */
  return ZSTD_decompressMultiFrame(dctx, dst, dstCapacity, src, srcSize, NULL,
                                   0, ddict);
}

DEVICE_INLINE size_t ZSTD_decompressDCtx(ZSTD_DCtx *dctx, void *dst,
                                         size_t dstCapacity, const void *src,
                                         size_t srcSize) {
  return ZSTD_decompress_usingDDict(dctx, dst, dstCapacity, src, srcSize,
                                    ZSTD_getDDict(dctx));
}

/******* DECOMPRESSION FUNCTIONS **********************************************/
/// Zstandard decompression functions.
/// `dst` must point to a space at least as large as the reconstructed output.
/// \note HIP/AMD: We only consider stack mode
DEVICE_INLINE size_t ZSTD_decompress(void *dst, size_t dstCapacity,
                                     const void *src, size_t srcSize) {

#if 0 && defined(ZSTD_HEAPMODE) && (ZSTD_HEAPMODE >= 1)
  size_t regenSize;
  ZSTD_DCtx *const dctx = ZSTD_createDCtx_internal(ZSTD_defaultCMem);
  RETURN_ERROR_IF(dctx == NULL, memory_allocation, "NULL pointer!");
  regenSize = ZSTD_decompressDCtx(dctx, dst, dstCapacity, src, srcSize);
  ZSTD_freeDCtx(dctx);
  return regenSize;
#else /* stack mode */
  __shared__ __align__(16) ZSTD_DCtx _dctx;
  ZSTD_DCtx *dctx = &_dctx;

  DEBUGLOG(5, " ZSTD_decompress: call ZSTD_initDCtx_internal");
  ZSTD_initDCtx_internal(dctx);
  DEBUGLOG(5, " ZSTD_decompress: enter ZSTD_decompressDCtx");
  return ZSTD_decompressDCtx(dctx, dst, dstCapacity, src, srcSize);
#endif
}

/*! ZSTD_getDictID_fromDict() :
 *  Provides the dictID stored within dictionary.
 *  if @return == 0, the dictionary is not conformant with Zstandard
 * specification. It can still be loaded, but as a content-only dictionary. */
DEVICE_INLINE unsigned ZSTD_getDictID_fromDict(const void *dict,
                                               size_t dictSize) {
  if (dictSize < 8)
    return 0;
  if (MEM_readLE32(dict) != ZSTD_MAGIC_DICTIONARY)
    return 0;
  return MEM_readLE32((const char *)dict + ZSTD_FRAMEIDSIZE);
}

/*! ZSTD_getDictID_fromFrame() :
 *  Provides the dictID required to decompress frame stored within `src`.
 *  If @return == 0, the dictID could not be decoded.
 *  This could for one of the following reasons :
 *  - The frame does not require a dictionary (most common case).
 *  - The frame was built with dictID intentionally removed.
 *    Needed dictionary is a hidden piece of information.
 *    Note : this use case also happens when using a non-conformant dictionary.
 *  - `srcSize` is too small, and as a result, frame header could not be
 * decoded. Note : possible if `srcSize < ZSTD_FRAMEHEADERSIZE_MAX`.
 *  - This is not a Zstandard frame.
 *  When identifying the exact failure cause, it's possible to use
 *  ZSTD_getFrameHeader(), which will provide a more precise error code. */
DEVICE_INLINE unsigned ZSTD_getDictID_fromFrame(const void *src,
                                                size_t srcSize) {
  ZSTD_FrameHeader zfp = {0, 0, 0, ZSTD_frame, 0, 0, 0, 0, 0};
  size_t const hError = ZSTD_getFrameHeader(&zfp, src, srcSize);
  if (ZSTD_isError(hError))
    return 0;
  return zfp.dictID;
}

} // namespace zstd
} // namespace hipcomp
