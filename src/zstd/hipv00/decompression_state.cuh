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

#include "common.cuh"
#include "xxhash.cuh"

namespace hipcomp {
namespace zstd {

/*-*************************************
 *  shared macros
 ***************************************/
#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define BOUNDED(min, val, max) (MAX(min, MIN(val, max)))

// from lib/common/zstd_internal.h

/*-*************************************
 *  Common constants
 ***************************************/
#define ZSTD_OPT_NUM (1 << 12)

#define ZSTD_REP_NUM 3 /* number of repcodes */
DEVICE_CONSTANT const U32 repStartValue[ZSTD_REP_NUM] = {1, 4, 8};

#define KB *(1 << 10)
#define MB *(1 << 20)
#define GB *(1U << 30)

#define BIT7 128
#define BIT6 64
#define BIT5 32
#define BIT4 16
#define BIT1 2
#define BIT0 1

#define ZSTD_WINDOWLOG_ABSOLUTEMIN 10
DEVICE_CONSTANT const size_t ZSTD_fcs_fieldSize[4] = {0, 2, 4, 8};
DEVICE_CONSTANT const size_t ZSTD_did_fieldSize[4] = {0, 1, 2, 4};

#define ZSTD_FRAMEIDSIZE 4 /* magic number size */

#define ZSTD_BLOCKHEADERSIZE                                                   \
  3 /* C standard doesn't allow `static const` variable to be init using       \
       another `static const` variable */
static constexpr size_t ZSTD_blockHeaderSize = ZSTD_BLOCKHEADERSIZE;
typedef enum { bt_raw, bt_rle, bt_compressed, bt_reserved } blockType_e;

#define ZSTD_FRAMECHECKSUMSIZE 4

#define MIN_SEQUENCES_SIZE 1 /* nbSeq==0 */
#define MIN_CBLOCK_SIZE                                                        \
  (1 /*litCSize*/ + 1 /* RLE or RAW */) /* for a non-null block */
#define MIN_LITERALS_FOR_4_STREAMS 6

typedef enum {
  set_basic,
  set_rle,
  set_compressed,
  set_repeat
} SymbolEncodingType_e;

#define LONGNBSEQ 0x7F00

#define MINMATCH 3

#define Litbits 8
#define LitHufLog 11
#define MaxLit ((1 << Litbits) - 1)
#define MaxML 52
#define MaxLL 35
#define DefaultMaxOff 28
#define MaxOff 31
#define MaxSeq MAX(MaxLL, MaxML) /* Assumption : MaxOff < MaxLL,MaxML */
#define MLFSELog 9
#define LLFSELog 9
#define OffFSELog 8
#define MaxFSELog MAX(MAX(MLFSELog, LLFSELog), OffFSELog)
#define MaxMLBits 16
#define MaxLLBits 16

#define ZSTD_MAX_HUF_HEADER_SIZE                                               \
  128 /* header + <= 127 byte tree description                                 \
       */
/* Each table cannot take more than #symbols * FSELog bits */
#define ZSTD_MAX_FSE_HEADERS_SIZE                                              \
  (((MaxML + 1) * MLFSELog + (MaxLL + 1) * LLFSELog +                          \
    (MaxOff + 1) * OffFSELog + 7) /                                            \
   8)

// Memory management related

// NOTE(HIP/AMD)
//   The original was set to 1<<16 (64 KiB), we reduced this to 32 KiB
//   so that litExtraBuffer fits into LDS for the majority of AMD GPU
//   architectures.
// TODO(HIP/AMD): Tune this according to architecture
#ifndef ZSTD_DECODER_INTERNAL_BUFFER
// #define ZSTD_DECODER_INTERNAL_BUFFER (1 << 16)
#define ZSTD_DECODER_INTERNAL_BUFFER (32 << 10)
#endif

#define ZSTD_LBMIN 64
// NOTE(HIP/AMD): Is only used in ZSTD_LITBUFFEREXTRASIZE computation as upper
// bound
#define ZSTD_LBMAX (128 << 10)

/* extra buffer, compensates when dst is not large enough to store litBuffer */
#define ZSTD_LITBUFFEREXTRASIZE                                                \
  BOUNDED(ZSTD_LBMIN, ZSTD_DECODER_INTERNAL_BUFFER, ZSTD_LBMAX)

// end from lib/common/zstd_internal.h

/**
 *  The minimum workspace size for the `workSpace` used in
 *  HUF_readDTableX1_wksp() and HUF_readDTableX2_wksp().
 *
 *  The space used depends on HUF_TABLELOG_MAX, ranging from ~1500 bytes when
 *  HUF_TABLE_LOG_MAX=12 to ~1850 bytes when HUF_TABLE_LOG_MAX=15.
 *  Buffer overflow errors may potentially occur if code modifications result in
 *  a required workspace size greater than that specified in the following
 *  macro.
 */
#define HUF_DECOMPRESS_WORKSPACE_SIZE ((2 << 10) + (1 << 9))
#define HUF_DECOMPRESS_WORKSPACE_SIZE_U32                                      \
  (HUF_DECOMPRESS_WORKSPACE_SIZE / sizeof(U32))

#define WILDCOPY_OVERLENGTH 32
#define WILDCOPY_VECLEN 16

#define ZSTD_FRAMEHEADERSIZE_MAX 18 /* can be useful for static allocation */

// Types

typedef long long __m128i __attribute__((__vector_size__(16), __may_alias__));

typedef long long __m128i_u
    __attribute__((__vector_size__(16), __may_alias__, __aligned__(1)));

DEVICE_INLINE __m128i _mm_loadu_si128(__m128i_u const *__P) { return *__P; }

DEVICE_INLINE void _mm_storeu_si128(__m128i_u *__P, __m128i __B) { *__P = __B; }

typedef struct ZSTD_DCtx_s ZSTD_DCtx;

typedef struct ZSTD_outBuffer_s {
  void *dst;
  size_t size;
  size_t pos;
} ZSTD_outBuffer;

typedef struct ZSTD_DDict_s ZSTD_DDict;

typedef enum {
  ZSTD_f_zstd1 = 0,
  ZSTD_f_zstd1_magicless = 1

} ZSTD_format_e;

typedef enum {

  ZSTD_d_validateChecksum = 0,
  ZSTD_d_ignoreChecksum = 1
} ZSTD_forceIgnoreChecksum_e;

typedef enum {

  ZSTD_rmd_refSingleDDict = 0,
  ZSTD_rmd_refMultipleDDicts = 1
} ZSTD_refMultipleDDicts_e;

typedef enum { ZSTD_frame, ZSTD_skippableFrame } ZSTD_FrameType_e;

typedef struct {
  unsigned long long frameContentSize;
  unsigned long long windowSize;
  unsigned blockSizeMax;
  ZSTD_FrameType_e frameType;
  unsigned headerSize;
  unsigned dictID;
  unsigned checksumFlag;
  unsigned _reserved1;
  unsigned _reserved2;
} ZSTD_FrameHeader;

typedef uint8_t U8;

typedef int16_t S16;

typedef U32 HUF_DTable;

typedef unsigned long long ZSTD_TraceCtx;

static DEVICE_CONSTANT const U8 LL_bits[MaxLL + 1] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  1,
    1, 1, 2, 2, 3, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

static DEVICE_CONSTANT const U8 ML_bits[MaxML + 1] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  1,  1,  1, 1,
    2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

DEVICE_INLINE void ZSTD_copy8(void *dst, const void *src) {
  __builtin_memcpy((dst), (src), (8));
}
#define COPY8(d, s)                                                            \
  do {                                                                         \
    ZSTD_copy8(d, s);                                                          \
    d += 8;                                                                    \
    s += 8;                                                                    \
  } while (0)

DEVICE_INLINE void ZSTD_copy16(void *dst, const void *src) {
  _mm_storeu_si128((__m128i *)dst, _mm_loadu_si128((const __m128i *)src));
}
#define COPY16(d, s)                                                           \
  do {                                                                         \
    ZSTD_copy16(d, s);                                                         \
    d += 16;                                                                   \
    s += 16;                                                                   \
  } while (0)

typedef enum {
  ZSTD_no_overlap,
  ZSTD_overlap_src_before_dst

} ZSTD_overlap_e;

/*! ZSTD_wildcopy() :
 *  Custom version of ZSTD_memcpy(), can over read/write up to
 * WILDCOPY_OVERLENGTH bytes (if length==0)
 *  @param ovtype controls the overlap detection
 *         - ZSTD_no_overlap: The source and destination are guaranteed to be at
 * least WILDCOPY_VECLEN bytes apart.
 *         - ZSTD_overlap_src_before_dst: The src and dst may overlap, but they
 * MUST be at least 8 bytes apart. The src buffer must be before the dst buffer.
 */
DEVICE_INLINE __attribute__((always_inline)) void
ZSTD_wildcopy(void *dst, const void *src, size_t length,
              ZSTD_overlap_e const ovtype) {
  ptrdiff_t diff = (BYTE *)dst - (const BYTE *)src;
  const BYTE *ip = (const BYTE *)src;
  BYTE *op = (BYTE *)dst;
  BYTE *const oend = op + length;

  if (ovtype == ZSTD_overlap_src_before_dst && diff < WILDCOPY_VECLEN) {
    /* Handle short offset copies. */
    do {
      COPY8(op, ip);
    } while (op < oend);
  } else {
    /* Separate out the first COPY16() call because the copy length is
     * almost certain to be short, so the branches have different
     * probabilities. Since it is almost certain to be short, only do
     * one COPY16() in the first call. Then, do two calls per loop since
     * at that point it is more likely to have a high trip count.
     */
    ZSTD_copy16(op, ip);
    if (16 >= length)
      return;
    op += 16;
    ip += 16;
    do {
      COPY16(op, ip);
      COPY16(op, ip);
    } while (op < oend);
  }
}

DEVICE_INLINE size_t ZSTD_limitCopy(void *dst, size_t dstCapacity,
                                    const void *src, size_t srcSize) {
  size_t const length = ((dstCapacity) < (srcSize) ? (dstCapacity) : (srcSize));
  if (length > 0) {
    __builtin_memcpy((dst), (src), (length));
  }
  return length;
}

typedef enum { ZSTD_bm_buffered = 0, ZSTD_bm_stable = 1 } ZSTD_bufferMode_e;

typedef struct {
  blockType_e blockType;
  U32 lastBlock;
  U32 origSize;
} blockProperties_t;

DEVICE_INLINE int ZSTD_cpuSupportsBmi2(void) {
  return 0; // NOTE(HIP/AMD): BMI2 only relevant for compression as of ZSTD
            // V1.5.7
}

static DEVICE_CONSTANT const U32 LL_base[MaxLL + 1] = {
    0,     1,     2,     3,     4,      5,      6,      7,      8,
    9,     10,    11,    12,    13,     14,     15,     16,     18,
    20,    22,    24,    28,    32,     40,     48,     64,     0x80,
    0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000};

static DEVICE_CONSTANT const U32 OF_base[MaxOff + 1] = {
    0,          1,         1,         5,         0xD,       0x1D,
    0x3D,       0x7D,      0xFD,      0x1FD,     0x3FD,     0x7FD,
    0xFFD,      0x1FFD,    0x3FFD,    0x7FFD,    0xFFFD,    0x1FFFD,
    0x3FFFD,    0x7FFFD,   0xFFFFD,   0x1FFFFD,  0x3FFFFD,  0x7FFFFD,
    0xFFFFFD,   0x1FFFFFD, 0x3FFFFFD, 0x7FFFFFD, 0xFFFFFFD, 0x1FFFFFFD,
    0x3FFFFFFD, 0x7FFFFFFD};

static DEVICE_CONSTANT const U8 OF_bits[MaxOff + 1] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31};

static DEVICE_CONSTANT const U32 ML_base[MaxML + 1] = {
    3,     4,     5,     6,      7,      8,      9,      10,     11,
    12,    13,    14,    15,     16,     17,     18,     19,     20,
    21,    22,    23,    24,     25,     26,     27,     28,     29,
    30,    31,    32,    33,     34,     35,     37,     39,     41,
    43,    47,    51,    59,     67,     83,     99,     0x83,   0x103,
    0x203, 0x403, 0x803, 0x1003, 0x2003, 0x4003, 0x8003, 0x10003};

typedef struct {
  U16 nextState;
  BYTE nbAdditionalBits;
  BYTE nbBits;
  U32 baseValue;
} ZSTD_seqSymbol;

#define SEQSYMBOL_TABLE_SIZE(log) (1 + (1 << (log)))

#define ZSTD_BUILD_FSE_TABLE_WKSP_SIZE                                         \
  (sizeof(S16) * (MaxSeq + 1) + (1u << MaxFSELog) + sizeof(U64))
#define ZSTD_BUILD_FSE_TABLE_WKSP_SIZE_U32                                     \
  ((ZSTD_BUILD_FSE_TABLE_WKSP_SIZE + sizeof(U32) - 1) / sizeof(U32))
#define ZSTD_HUFFDTABLE_CAPACITY_LOG 12

#define HUF_DTABLE_SIZE(maxTableLog) (1 + (1 << (maxTableLog)))

typedef struct {
  ZSTD_seqSymbol LLTable[SEQSYMBOL_TABLE_SIZE(
      LLFSELog)]; /* Note : Space reserved for FSE Tables */
  ZSTD_seqSymbol OFTable[SEQSYMBOL_TABLE_SIZE(
      OffFSELog)]; /* is also used as temporary workspace while building
                      hufTable during DDict creation */
  ZSTD_seqSymbol MLTable[SEQSYMBOL_TABLE_SIZE(
      MLFSELog)]; /* and therefore must be at least
                     HUF_DECOMPRESS_WORKSPACE_SIZE large */
  HUF_DTable hufTable[HUF_DTABLE_SIZE(
      ZSTD_HUFFDTABLE_CAPACITY_LOG)]; /* can accommodate HUF_decompress4X */
  U32 rep[ZSTD_REP_NUM];
  U32 workspace[ZSTD_BUILD_FSE_TABLE_WKSP_SIZE_U32];
} ZSTD_entropyDTables_t;

typedef enum {
  ZSTDds_getFrameHeaderSize,
  ZSTDds_decodeFrameHeader,
  ZSTDds_decodeBlockHeader,
  ZSTDds_decompressBlock,
  ZSTDds_decompressLastBlock,
  ZSTDds_checkChecksum,
  ZSTDds_decodeSkippableHeader,
  ZSTDds_skipFrame
} ZSTD_dStage;

typedef enum {
  zdss_init = 0,
  zdss_loadHeader,
  zdss_read,
  zdss_load,
  zdss_flush
} ZSTD_dStreamStage;

typedef enum {
  ZSTD_use_indefinitely = -1,
  ZSTD_dont_use = 0,
  ZSTD_use_once = 1
} ZSTD_dictUses_e;

typedef struct {
  const ZSTD_DDict **ddictPtrTable;
  size_t ddictPtrTableSize;
  size_t ddictPtrCount;
} ZSTD_DDictHashSet;

typedef enum {
  ZSTD_not_in_dst = 0, /* Stored entirely within litExtraBuffer */
  ZSTD_in_dst =
      1, /* Stored entirely within dst (in memory after current output write) */
  ZSTD_split = 2 /* Split between litExtraBuffer and dst */
} ZSTD_litLocation_e;

// TODO(HIP/AMD): becomes template when we add prefetch functionality
struct ZSTD_DCtx_s {
  const ZSTD_seqSymbol *LLTptr;
  const ZSTD_seqSymbol *MLTptr;
  const ZSTD_seqSymbol *OFTptr;
  const HUF_DTable *HUFptr;
  ZSTD_entropyDTables_t entropy;
  //: NOTE(HIP/AMD): expected sizeof(workspace): 640 bytes
  U32 workspace[HUF_DECOMPRESS_WORKSPACE_SIZE_U32]; /* space needed when
                                                       building huffman tables
                                                     */
  /* window/prefix management */
  const void *previousDstEnd; /* detect continuity */
  const void *prefixStart;    /* start of current segment */
  const void *virtualStart; /* virtual start of previous segment if it was just
                               before current one */
  const void *dictEnd;      /* end of previous segment */
  size_t expected;
  ZSTD_FrameHeader fParams;
  U64 processedCSize;
  U64 decodedSize;
  blockType_e
      bType; /* used in ZSTD_decompressContinue(), store blockType between block
                header decoding and block decompression stages */
  ZSTD_dStage stage;
  U32 litEntropy;
  U32 fseEntropy;
  XXH64_state_t xxhState;
  size_t headerSize;
  ZSTD_format_e format;
  ZSTD_forceIgnoreChecksum_e
      forceIgnoreChecksum; /* User specified: if == 1, will ignore checksums in
                              compressed frame. Default == 0 */
  U32 validateChecksum;    /* if == 1, will validate checksum. Is == 1 if
                              (fParams.checksumFlag == 1) and (forceIgnoreChecksum
                              == 0). */
  const BYTE *litPtr;
  ZSTD_customMem customMem;
  size_t litSize;
  size_t rleSize;
  size_t staticSize;
  int isFrameDecompression;

#if 0
#if DYNAMIC_BMI2
  int bmi2;                     /* == 1 if the CPU supports BMI2 and 0 otherwise. CPU support is determined dynamically once per context lifetime. */
#endif
#endif //: NOTE(HIP/AMD): BMI2 only relevant for compression as of ZSTD V1.5.7

  /* dictionary */
  ZSTD_DDict *ddictLocal;
  const ZSTD_DDict *
      ddict; /* set by ZSTD_initDStream_usingDDict(), or ZSTD_DCtx_refDDict() */
  U32 dictID;
  int ddictIsCold; /* if == 1 : dictionary is "new" for working context, and
                      presumed "cold" (not in cpu cache) */
  ZSTD_dictUses_e dictUses;
  ZSTD_DDictHashSet *ddictSet; /* Hash set for multiple ddicts */
  ZSTD_refMultipleDDicts_e
      refMultipleDDicts; /* User specified: if == 1, will allow references to
                            multiple DDicts. Default == 0 (disabled) */
  int disableHufAsm;
  int maxBlockSizeParam;

  /* streaming */
#if 0
    ZSTD_dStreamStage streamStage;
#endif //: NOTE(HIP/AMD): we do not support streaming
  char *inBuff;
  size_t inBuffSize;
  size_t inPos;
  size_t maxWindowSize;
  char *outBuff;
  size_t outBuffSize;
  size_t outStart;
  size_t outEnd;
  size_t lhSize;
#if defined(ZSTD_LEGACY_SUPPORT) && (ZSTD_LEGACY_SUPPORT >= 1)
  void *legacyContext;
  U32 previousLegacyVersion;
  U32 legacyVersion;
#endif
  U32 hostageByte;
  int noForwardProgress;
  ZSTD_bufferMode_e outBufferMode;
  ZSTD_outBuffer expectedOutBuffer;

  /* workspace */
  BYTE *litBuffer;
  const BYTE *litBufferEnd;
  ZSTD_litLocation_e litBufferLocation;
  //: NOTE(HIP/AMD): with default 64bit settings, expected
  //: sizeof(litExtraBuffer): 65568 = 64 KiB
  BYTE litExtraBuffer[ZSTD_LITBUFFEREXTRASIZE +
                      WILDCOPY_OVERLENGTH]; /* literal buffer can be split
                                               between storage within dst and
                                               within this scratch buffer */
  //: NOTE(HIP/AMD): expected sizeof(headerBuffer): 18 bytes
  BYTE headerBuffer[ZSTD_FRAMEHEADERSIZE_MAX];
  size_t oversizedDuration;

#if 0
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
    void const* dictContentBeginForFuzzing;
    void const* dictContentEndForFuzzing;
#endif
#endif //: NOTE(HIP/AMD): we do not support fuzzing
       /* Tracing */
#if ZSTD_TRACE
  ZSTD_TraceCtx traceCtx;
#endif
}; /* typedef'd to ZSTD_DCtx within "zstd.h" */

DEVICE_INLINE int ZSTD_DCtx_get_bmi2(const struct ZSTD_DCtx_s *dctx) {
  return 0; //: NOTE(HIP/AMD): BMI2 only relevant for compression as of ZSTD
            //: V1.5.7
}

typedef enum { not_streaming = 0, is_streaming = 1 } streaming_operation;

} // namespace zstd
} // namespace hipcomp
