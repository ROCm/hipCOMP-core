/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

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

#include "assert.h"
#include "hip/hip_runtime.h"

// from lib/common/{debug}.h

// TODO(HIP/AMD): Unify debug infrastructure
#ifdef HIPCOMP_DEBUG_OUTPUT
#define DEBUGLEVEL 5
#endif

#if (DEBUGLEVEL >= 2)
// NOTE(HIP/AMD): device printf
#define ZSTD_DEBUG_PRINT(...) printf(__VA_ARGS__)

constexpr __device__ const int g_debuglevel = DEBUGLEVEL; /* the variable is
                            only declared, it actually lives in debug.c, and is
                            shared by the whole process. It's not thread-safe.
                            It's useful when enabling very verbose levels
                            on selective conditions (such as position in src) */

#define RAWLOG(l, ...)                                                         \
  do {                                                                         \
    if (l <= g_debuglevel) {                                                   \
      ZSTD_DEBUG_PRINT(__VA_ARGS__);                                           \
    }                                                                          \
  } while (0)

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define LINE_AS_STRING TOSTRING(__LINE__)

#define DEBUGLOG(l, ...)                                                       \
  do {                                                                         \
    if (l <= g_debuglevel) {                                                   \
      ZSTD_DEBUG_PRINT(__FILE__ ":" LINE_AS_STRING ": " __VA_ARGS__);          \
      ZSTD_DEBUG_PRINT(" \n");                                                 \
    }                                                                          \
  } while (0)
#else
#define RAWLOG(l, ...)                                                         \
  do {                                                                         \
  } while (0) /* disabled */
#define DEBUGLOG(l, ...)                                                       \
  do {                                                                         \
  } while (0) /* disabled */
#endif

// DEBUG / ASSERT functionality
/* static assert is triggered at compile time, leaving no runtime artefact.
 * static assert only works with compile-time constants.
 * Also, this variant can only be used inside a function. */
#define DEBUG_STATIC_ASSERT(c) (void)sizeof(char[(c) ? 1 : -1])

/* ---- static assert (debug) --- */
#define ZSTD_STATIC_ASSERT(c) DEBUG_STATIC_ASSERT(c)
#define ZSTD_isError ERR_isError /* for inlining */

// end from lib/common/{debug}.h

#define NO_PREFETCH 1 //: NOTE(HIP/AMD) we disable prefetching macros for now

// from lib/common/compiler.h

// NOTE(HIP/AMD): We use only the clang specialization here.
//                More general handling in
#define ZSTD_ALLOW_POINTER_OVERFLOW_ATTR                                       \
  __attribute__((no_sanitize("pointer-overflow")))

/* C-language Attributes are added in C23. */
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ > 201710L) &&               \
    defined(__has_c_attribute)
#define ZSTD_HAS_C_ATTRIBUTE(x) __has_c_attribute(x)
#else
#define ZSTD_HAS_C_ATTRIBUTE(x) 0
#endif

/* Only use C++ attributes in C++. Some compilers report support for C++
 * attributes when compiling with C.
 */
#if defined(__cplusplus) && defined(__has_cpp_attribute)
#define ZSTD_HAS_CPP_ATTRIBUTE(x) __has_cpp_attribute(x)
#else
#define ZSTD_HAS_CPP_ATTRIBUTE(x) 0
#endif

/* Define ZSTD_FALLTHROUGH macro for annotating switch case with the
 * 'fallthrough' attribute.
 * - C23: https://en.cppreference.com/w/c/language/attributes/fallthrough
 * - CPP17: https://en.cppreference.com/w/cpp/language/attributes/fallthrough
 * - Else: __attribute__((__fallthrough__))
 */
#ifndef ZSTD_FALLTHROUGH
#if ZSTD_HAS_C_ATTRIBUTE(fallthrough)
#define ZSTD_FALLTHROUGH [[fallthrough]]
#elif ZSTD_HAS_CPP_ATTRIBUTE(fallthrough)
#define ZSTD_FALLTHROUGH [[fallthrough]]
#elif __has_attribute(__fallthrough__)
/* Leading semicolon is to satisfy gcc-11 with -pedantic. Without the semicolon
 * gcc complains about: a label can only be part of a statement and a
 * declaration is not a statement.
 */
#define ZSTD_FALLTHROUGH                                                       \
  ;                                                                            \
  __attribute__((__fallthrough__))
#else
#define ZSTD_FALLTHROUGH
#endif
#endif

// NOTE(HIP/AMD): We disable prefetching
#ifndef NO_PREFETCH
#define NO_PREFETCH
#endif

// TODO(HIP/AMD): Pick only relevant branch
/* prefetch
 * can be disabled, by declaring NO_PREFETCH build macro */
#if defined(NO_PREFETCH)
#define PREFETCH_L1(ptr)                                                       \
  do {                                                                         \
    (void)(ptr);                                                               \
  } while (0) /* disabled */
#define PREFETCH_L2(ptr)                                                       \
  do {                                                                         \
    (void)(ptr);                                                               \
  } while (0) /* disabled */
#else
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_I86)) &&               \
    !defined(                                                                  \
        _M_ARM64EC) /* _mm_prefetch() is not defined outside of x86/x64 */
#include <mmintrin.h> /* https://msdn.microsoft.com/fr-fr/library/84szxsww(v=vs.90).aspx */
#define PREFETCH_L1(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T0)
#define PREFETCH_L2(ptr) _mm_prefetch((const char *)(ptr), _MM_HINT_T1)
#elif defined(__GNUC__) &&                                                     \
    ((__GNUC__ >= 4) || ((__GNUC__ == 3) && (__GNUC_MINOR__ >= 1)))
#define PREFETCH_L1(ptr)                                                       \
  __builtin_prefetch((ptr), 0 /* rw==read */, 3 /* locality */)
#define PREFETCH_L2(ptr)                                                       \
  __builtin_prefetch((ptr), 0 /* rw==read */, 2 /* locality */)
#elif defined(__aarch64__)
#define PREFETCH_L1(ptr)                                                       \
  do {                                                                         \
    __asm__ __volatile__("prfm pldl1keep, %0" ::"Q"(*(ptr)));                  \
  } while (0)
#define PREFETCH_L2(ptr)                                                       \
  do {                                                                         \
    __asm__ __volatile__("prfm pldl2keep, %0" ::"Q"(*(ptr)));                  \
  } while (0)
#else
#define PREFETCH_L1(ptr)                                                       \
  do {                                                                         \
    (void)(ptr);                                                               \
  } while (0) /* disabled */
#define PREFETCH_L2(ptr)                                                       \
  do {                                                                         \
    (void)(ptr);                                                               \
  } while (0) /* disabled */
#endif
#endif /* NO_PREFETCH */

#define CACHELINE_SIZE 64

#define PREFETCH_AREA(p, s)                                                    \
  do {                                                                         \
    const char *const _ptr = (const char *)(p);                                \
    size_t const _size = (size_t)(s);                                          \
    size_t _pos;                                                               \
    for (_pos = 0; _pos < _size; _pos += CACHELINE_SIZE) {                     \
      PREFETCH_L2(_ptr + _pos);                                                \
    }                                                                          \
  } while (0)

// end from lib/common/compiler.h

// HIP modifiers/definitions

#define DEVICE __device__
#define DEVICE_INLINE __device__ inline
#define DEVICE_CONSTANT __device__ __constant__

#define assert_hip_amd(a)                                                      \
  assert(a) // indicate that an assertion was introduced for the GPU code

namespace hipcomp {
namespace zstd {

// from zstd.h:

/* *************************************
 *  Constants
 ***************************************/

/* All magic numbers are supposed read/written to/from files/memory using
 * little-endian convention */
#define ZSTD_MAGICNUMBER 0xFD2FB528      /* valid since v0.8.0 */
#define ZSTD_MAGIC_DICTIONARY 0xEC30A437 /* valid since v0.7.0 */
#define ZSTD_MAGIC_SKIPPABLE_START                                             \
  0x184D2A50 /* all 16 values, from 0x184D2A50 to 0x184D2A5F, signal the       \
                beginning of a skippable frame */
#define ZSTD_MAGIC_SKIPPABLE_MASK 0xFFFFFFF0

#define ZSTD_BLOCKSIZELOG_MAX 17
#define ZSTD_BLOCKSIZE_MAX (1 << ZSTD_BLOCKSIZELOG_MAX)

/****************************************************************************************
 *   experimental API (static linking only)
 ****************************************************************************************
 * The following symbols and constants
 * are not planned to join "stable API" status in the near future.
 * They can still change in future versions.
 * Some of them are planned to remain in the static_only section indefinitely.
 * Some of them might be removed in the future (especially when redundant with
 *existing stable functions)
 * ***************************************************************************************/

#define ZSTD_FRAMEHEADERSIZE_PREFIX(format)                                    \
  ((format) == ZSTD_f_zstd1                                                    \
       ? 5                                                                     \
       : 1) /* minimum input size required to query frame header size */
#define ZSTD_FRAMEHEADERSIZE_MIN(format) ((format) == ZSTD_f_zstd1 ? 6 : 2)
#define ZSTD_FRAMEHEADERSIZE_MAX 18 /* can be useful for static allocation */
#define ZSTD_SKIPPABLEHEADERSIZE 8

/* compression parameter bounds */
#define ZSTD_WINDOWLOG_MAX_32 30
#define ZSTD_WINDOWLOG_MAX_64 31
#define ZSTD_WINDOWLOG_MAX                                                     \
  ((int)(sizeof(size_t) == 4 ? ZSTD_WINDOWLOG_MAX_32 : ZSTD_WINDOWLOG_MAX_64))
#define ZSTD_WINDOWLOG_MIN 10
#define ZSTD_HASHLOG_MAX ((ZSTD_WINDOWLOG_MAX < 30) ? ZSTD_WINDOWLOG_MAX : 30)
#define ZSTD_HASHLOG_MIN 6
#define ZSTD_CHAINLOG_MAX_32 29
#define ZSTD_CHAINLOG_MAX_64 30
#define ZSTD_CHAINLOG_MAX                                                      \
  ((int)(sizeof(size_t) == 4 ? ZSTD_CHAINLOG_MAX_32 : ZSTD_CHAINLOG_MAX_64))
#define ZSTD_CHAINLOG_MIN ZSTD_HASHLOG_MIN
#define ZSTD_SEARCHLOG_MAX (ZSTD_WINDOWLOG_MAX - 1)
#define ZSTD_SEARCHLOG_MIN 1
#define ZSTD_MINMATCH_MAX                                                      \
  7 /* only for ZSTD_fast, other strategies are limited to 6 */
#define ZSTD_MINMATCH_MIN                                                      \
  3 /* only for ZSTD_btopt+, faster strategies are limited to 4 */
#define ZSTD_TARGETLENGTH_MAX ZSTD_BLOCKSIZE_MAX
#define ZSTD_TARGETLENGTH_MIN                                                  \
  0 /* note : comparing this constant to an unsigned results in a tautological \
       test */
#define ZSTD_STRATEGY_MIN ZSTD_fast
#define ZSTD_STRATEGY_MAX ZSTD_btultra2
#define ZSTD_BLOCKSIZE_MAX_MIN                                                 \
  (1 << 10) /* The minimum valid max blocksize. Maximum blocksizes smaller     \
               than this make compressBound() inaccurate. */

#define ZSTD_OVERLAPLOG_MIN 0
#define ZSTD_OVERLAPLOG_MAX 9

#define ZSTD_WINDOWLOG_LIMIT_DEFAULT                                           \
  27 /* by default, the streaming decoder will refuse any frame                \
      * requiring larger than (1<<ZSTD_WINDOWLOG_LIMIT_DEFAULT) window size,   \
      * to preserve host's memory from unreasonable requirements.              \
      * This limit can be overridden using                                     \
      * ZSTD_DCtx_setParameter(,ZSTD_d_windowLogMax,). The limit does not      \
      * apply for one-pass decoders (such as ZSTD_decompress()), since no      \
      * additional memory is allocated */

/* LDM parameter bounds */
#define ZSTD_LDM_HASHLOG_MIN ZSTD_HASHLOG_MIN
#define ZSTD_LDM_HASHLOG_MAX ZSTD_HASHLOG_MAX
#define ZSTD_LDM_MINMATCH_MIN 4
#define ZSTD_LDM_MINMATCH_MAX 4096
#define ZSTD_LDM_BUCKETSIZELOG_MIN 1
#define ZSTD_LDM_BUCKETSIZELOG_MAX 8
#define ZSTD_LDM_HASHRATELOG_MIN 0
#define ZSTD_LDM_HASHRATELOG_MAX (ZSTD_WINDOWLOG_MAX - ZSTD_HASHLOG_MIN)

/* Advanced parameter bounds */
#define ZSTD_TARGETCBLOCKSIZE_MIN                                              \
  1340 /* suitable to fit into an ethernet / wifi / 4G transport frame */
#define ZSTD_TARGETCBLOCKSIZE_MAX ZSTD_BLOCKSIZE_MAX
#define ZSTD_SRCSIZEHINT_MIN 0
#define ZSTD_SRCSIZEHINT_MAX INT_MAX

// end from zstd.h

// from lib/common/bitstream.h

#define STREAM_ACCUMULATOR_MIN_32 25
#define STREAM_ACCUMULATOR_MIN_64 57
#define STREAM_ACCUMULATOR_MIN                                                 \
  ((U32)(MEM_32bits() ? STREAM_ACCUMULATOR_MIN_32 : STREAM_ACCUMULATOR_MIN_64))

// end from lib/common/bitstream.h

// from other such as common.h, zstd_decompress_internal.(h|c):

/*-*************************************
 *  shared macros
 ***************************************/
#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define BOUNDED(min, val, max) (MAX(min, MIN(val, max)))

// Common types
typedef long unsigned int size_t;
typedef long int ptrdiff_t;

typedef unsigned char __uint8_t;
typedef signed short int __int16_t;
typedef unsigned int __uint32_t;
typedef unsigned long int __uint64_t;
typedef unsigned short int __uint16_t;

typedef __uint8_t uint8_t;
typedef __uint16_t uint16_t;
typedef __uint32_t uint32_t;
typedef __uint64_t uint64_t;
typedef __int16_t int16_t;

typedef uint8_t BYTE;
typedef uint16_t U16;
typedef uint32_t U32;
typedef uint64_t U64;
typedef int16_t S16;

typedef __attribute__((aligned(1))) U16 unalign16;
typedef __attribute__((aligned(1))) U32 unalign32;
typedef __attribute__((aligned(1))) U64 unalign64;

// from lib/{zstd_errors.h,common/error_private.h}

typedef enum {
  ZSTD_error_no_error = 0,
  ZSTD_error_GENERIC = 1,
  ZSTD_error_prefix_unknown = 10,
  ZSTD_error_version_unsupported = 12,
  ZSTD_error_frameParameter_unsupported = 14,
  ZSTD_error_frameParameter_windowTooLarge = 16,
  ZSTD_error_corruption_detected = 20,
  ZSTD_error_checksum_wrong = 22,
  ZSTD_error_literals_headerWrong = 24,
  ZSTD_error_dictionary_corrupted = 30,
  ZSTD_error_dictionary_wrong = 32,
  ZSTD_error_dictionaryCreation_failed = 34,
  ZSTD_error_parameter_unsupported = 40,
  ZSTD_error_parameter_combination_unsupported = 41,
  ZSTD_error_parameter_outOfBound = 42,
  ZSTD_error_tableLog_tooLarge = 44,
  ZSTD_error_maxSymbolValue_tooLarge = 46,
  ZSTD_error_maxSymbolValue_tooSmall = 48,
  ZSTD_error_cannotProduce_uncompressedBlock = 49,
  ZSTD_error_stabilityCondition_notRespected = 50,
  ZSTD_error_stage_wrong = 60,
  ZSTD_error_init_missing = 62,
  ZSTD_error_memory_allocation = 64,
  ZSTD_error_workSpace_tooSmall = 66,
  ZSTD_error_dstSize_tooSmall = 70,
  ZSTD_error_srcSize_wrong = 72,
  ZSTD_error_dstBuffer_null = 74,
  ZSTD_error_noForwardProgress_destFull = 80,
  ZSTD_error_noForwardProgress_inputEmpty = 82,

  ZSTD_error_frameIndex_tooLarge = 100,
  ZSTD_error_seekableIO = 102,
  ZSTD_error_dstBuffer_wrong = 104,
  ZSTD_error_srcBuffer_wrong = 105,
  ZSTD_error_sequenceProducer_failed = 106,
  ZSTD_error_externalSequences_invalid = 107,
  ZSTD_error_maxCode = 120
} ZSTD_ErrorCode;

#define ERROR(name) ((size_t)-ZSTD_error_##name)

/* ****************************************
 *  Compiler-specific
 ******************************************/
#define ERR_STATIC DEVICE_INLINE //: HIP/AMD: Redefinition

/*-****************************************
 *  Customization (error_public.h)
 ******************************************/
typedef ZSTD_ErrorCode ERR_enum;
#define PREFIX(name) ZSTD_error_##name

/*-****************************************
 *  Error codes handling
 ******************************************/
#undef ERROR /* already defined on Visual Studio */
#define ERROR(name) ZSTD_ERROR(name)
#define ZSTD_ERROR(name) ((size_t)-PREFIX(name))

DEVICE_INLINE unsigned ERR_isError(size_t code) {
  return (code > ERROR(maxCode));
}

DEVICE_INLINE ERR_enum ERR_getErrorCode(size_t code) {
  if (!ERR_isError(code))
    return (ERR_enum)0;
  return (ERR_enum)(0 - code);
}

/* check and forward error code */
#define CHECK_V_F(e, f)                                                        \
  size_t const e = f;                                                          \
  do {                                                                         \
    if (ERR_isError(e))                                                        \
      return e;                                                                \
  } while (0)
#define CHECK_F(f)                                                             \
  do {                                                                         \
    CHECK_V_F(_var_err__, f);                                                  \
  } while (0)

/*-****************************************
 *  Error Strings
 ******************************************/

// DEVICE_INLINE const char* ERR_getErrorString(ERR_enum code);   /*
// error_private.c */

DEVICE_INLINE const char *ERR_getErrorName(size_t code) {
  return "";
  // return ERR_getErrorString(ERR_getErrorCode(code));
}

/**
 * Ignore: this is an internal helper.
 *
 * This is a helper function to help force C99-correctness during compilation.
 * Under strict compilation modes, variadic macro arguments can't be empty.
 * However, variadic function arguments can be. Using a function therefore lets
 * us statically check that at least one (string) argument was passed,
 * independent of the compilation flags.
 */
DEVICE_INLINE
void _force_has_format_string(const char *format, ...) { (void)format; }

/**
 * Ignore: this is an internal helper.
 *
 * We want to force this function invocation to be syntactically correct, but
 * we don't want to force runtime evaluation of its arguments.
 */
#define _FORCE_HAS_FORMAT_STRING(...)                                          \
  do {                                                                         \
    if (0) {                                                                   \
      _force_has_format_string(__VA_ARGS__);                                   \
    }                                                                          \
  } while (0)

#define ERR_QUOTE(str) #str

/**
 * Return the specified error if the condition evaluates to true.
 *
 * In debug modes, prints additional information.
 * In order to do that (particularly, printing the conditional that failed),
 * this can't just wrap RETURN_ERROR().
 */
#define RETURN_ERROR_IF(cond, err, ...)                                        \
  do {                                                                         \
    if (cond) {                                                                \
      RAWLOG(3, "%s:%d: ERROR!: check %s failed, returning %s", __FILE__,      \
             __LINE__, ERR_QUOTE(cond), ERR_QUOTE(ERROR(err)));                \
      _FORCE_HAS_FORMAT_STRING(__VA_ARGS__);                                   \
      RAWLOG(3, ": " __VA_ARGS__);                                             \
      RAWLOG(3, "\n");                                                         \
      return ERROR(err);                                                       \
    }                                                                          \
  } while (0)

/**
 * Unconditionally return the specified error.
 *
 * In debug modes, prints additional information.
 */
#define RETURN_ERROR(err, ...)                                                 \
  do {                                                                         \
    RAWLOG(3, "%s:%d: ERROR!: unconditional check failed, returning %s",       \
           __FILE__, __LINE__, ERR_QUOTE(ERROR(err)));                         \
    _FORCE_HAS_FORMAT_STRING(__VA_ARGS__);                                     \
    RAWLOG(3, ": " __VA_ARGS__);                                               \
    RAWLOG(3, "\n");                                                           \
    return ERROR(err);                                                         \
  } while (0)

/**
 * If the provided expression evaluates to an error code, returns that error
 * code.
 *
 * In debug modes, prints additional information.
 */
#define FORWARD_IF_ERROR(err, ...)                                             \
  do {                                                                         \
    size_t const err_code = (err);                                             \
    if (ERR_isError(err_code)) {                                               \
      RAWLOG(3, "%s:%d: ERROR!: forwarding error in %s: %s", __FILE__,         \
             __LINE__, ERR_QUOTE(err), ERR_getErrorName(err_code));            \
      _FORCE_HAS_FORMAT_STRING(__VA_ARGS__);                                   \
      RAWLOG(3, ": " __VA_ARGS__);                                             \
      RAWLOG(3, "\n");                                                         \
      return err_code;                                                         \
    }                                                                          \
  } while (0)

// end from lib/{zstd_errors.h,common/error_private.h}

// from lib/common/zstd_common.c

/*! ZSTD_getError() :
 *  convert a `size_t` function result into a proper ZSTD_errorCode enum */
DEVICE_INLINE ZSTD_ErrorCode ZSTD_getErrorCode(size_t code) {
  return ERR_getErrorCode(code);
}

// end from lib/common/zstd_common.c

// from lib/common/zstd_internal.h

#define ZSTD_isError ERR_isError /* for inlining */
#define FSE_isError ERR_isError
#define HUF_isError ERR_isError

#define LL_DEFAULTNORMLOG 6 /* for static allocation */
constexpr const U32 LL_defaultNormLog = LL_DEFAULTNORMLOG;

#define ML_DEFAULTNORMLOG 6 /* for static allocation */
constexpr const U32 ML_defaultNormLog = ML_DEFAULTNORMLOG;

#define OF_DEFAULTNORMLOG 5 /* for static allocation */
constexpr const U32 OF_defaultNormLog = OF_DEFAULTNORMLOG;

// end from lib/common/zstd_internal.h

// Common functions
#define ZSTD_memcpy(d, s, n) __builtin_memcpy((d), (s), (n))
#define ZSTD_memmove(d, s, n) __builtin_memmove((d), (s), (n))
#define ZSTD_memset(d, s, n) __builtin_memset((d), (s), (n))

#define LIKELY(x) (__builtin_expect((x), 1))
#define UNLIKELY(x) (__builtin_expect((x), 0))

/* check and forward error code */
#define CHECK_V_F(e, f)                                                        \
  size_t const e = f;                                                          \
  do {                                                                         \
    if (ERR_isError(e))                                                        \
      return e;                                                                \
  } while (0)

#define CHECK_F(f)                                                             \
  do {                                                                         \
    CHECK_V_F(_var_err__, f);                                                  \
  } while (0)

DEVICE_INLINE unsigned MEM_32bits(void) { return sizeof(size_t) == 4; }

DEVICE_INLINE unsigned MEM_64bits(void) { return sizeof(size_t) == 8; }

constexpr DEVICE_INLINE unsigned MEM_isLittleEndian(void) { return 1; }

DEVICE_INLINE U16 MEM_read16(const void *ptr) {
  return *(const unalign16 *)ptr;
}

DEVICE_INLINE U32 MEM_read32(const void *ptr) {
  return *(const unalign32 *)ptr;
}

DEVICE_INLINE U64 MEM_read64(const void *ptr) {
  return *(const unalign64 *)ptr;
}

DEVICE_INLINE void MEM_write16(void *memPtr, U16 value) {
  *(unalign16 *)memPtr = value;
}

DEVICE_INLINE void MEM_write32(void *memPtr, U32 value) {
  *(unalign32 *)memPtr = value;
}

DEVICE_INLINE void MEM_write64(void *memPtr, U64 value) {
  *(unalign64 *)memPtr = value;
}

DEVICE_INLINE U32 MEM_swap32(U32 in) { return __builtin_bswap32(in); }

DEVICE_INLINE U64 MEM_swap64(U64 in) { return __builtin_bswap64(in); }

DEVICE_INLINE U16 MEM_readLE16(const void *memPtr) {
  //: if (MEM_isLittleEndian())
  return MEM_read16(memPtr);
}

DEVICE_INLINE U32 MEM_readLE24(const void *memPtr) {
  return (U32)MEM_readLE16(memPtr) + ((U32)(((const BYTE *)memPtr)[2]) << 16);
}

DEVICE_INLINE U32 MEM_readLE32(const void *memPtr) {
  //: if (MEM_isLittleEndian())
  return MEM_read32(memPtr);
}

DEVICE_INLINE void MEM_writeLE32(void *memPtr, U32 val32) {
  //: if (MEM_isLittleEndian())
  MEM_write32(memPtr, val32);
}

DEVICE_INLINE U64 MEM_readLE64(const void *memPtr) {
  //: if (MEM_isLittleEndian())
  return MEM_read64(memPtr);
}

DEVICE_INLINE void MEM_writeLE64(void *memPtr, U32 val32) {
  //: if (MEM_isLittleEndian())
  MEM_write64(memPtr, val32);
}

DEVICE_INLINE size_t MEM_readLEST(const void *memPtr) {
  //: if (MEM_32bits())
  //: else
  return (size_t)MEM_readLE64(memPtr);
}

DEVICE_INLINE void MEM_writeLEST(void *memPtr, size_t val) {
  //: if (MEM_32bits())
  //: else
  MEM_writeLE64(memPtr, (U64)val);
}

DEVICE_INLINE int ZSTD_isPower2(size_t u) { return (u & (u - 1)) == 0; }

DEVICE_INLINE
__attribute__((no_sanitize("pointer-overflow"))) ptrdiff_t
ZSTD_wrappedPtrDiff(unsigned char const *lhs, unsigned char const *rhs) {
  return lhs - rhs;
}

DEVICE_INLINE
__attribute__((no_sanitize("pointer-overflow"))) const void *
ZSTD_wrappedPtrAdd(const void *ptr, ptrdiff_t add) {
  return (const char *)ptr + add;
}

DEVICE_INLINE
__attribute__((no_sanitize("pointer-overflow"))) const void *
ZSTD_wrappedPtrSub(const void *ptr, ptrdiff_t sub) {
  return (const char *)ptr - sub;
}

DEVICE_INLINE void *ZSTD_maybeNullPtrAdd(void *ptr, ptrdiff_t add) {
  return add > 0 ? (char *)ptr + add : ptr;
}

DEVICE_INLINE unsigned ZSTD_countLeadingZeros32(U32 val) {
  return (unsigned)__builtin_clz(val);
}

DEVICE_INLINE unsigned ZSTD_countTrailingZeros32(U32 val) {
  return (unsigned)__builtin_ctz(val);
}

DEVICE_INLINE unsigned ZSTD_countTrailingZeros64(U64 val) {
  return (unsigned)__builtin_ctzll(val);
}

DEVICE_INLINE unsigned ZSTD_NbCommonBytes(size_t val) {
  // if (MEM_isLittleEndian()) {
  // if (MEM_64bits()) {
  return ZSTD_countTrailingZeros64((U64)val) >> 3;
}

DEVICE_INLINE unsigned ZSTD_highbit32(U32 val) {
  return 31 - ZSTD_countLeadingZeros32(val);
}

DEVICE_INLINE U64 ZSTD_rotateRight_U64(U64 const value, U32 count) {
  count &= 0x3F;
  return (value >> count) | (U64)(value << ((0U - count) & 0x3F));
}

DEVICE_INLINE U32 ZSTD_rotateRight_U32(U32 const value, U32 count) {
  count &= 0x1F;
  return (value >> count) | (U32)(value << ((0U - count) & 0x1F));
}

DEVICE_INLINE U16 ZSTD_rotateRight_U16(U16 const value, U32 count) {
  count &= 0x0F;
  return (value >> count) | (U16)(value << ((0U - count) & 0x0F));
}

// from zstd.h

/*! Custom memory allocation :
 *  These prototypes make it possible to pass your own allocation/free
 * functions. ZSTD_customMem is provided at creation time, using
 * ZSTD_create*_advanced() variants listed below. All allocation/free operations
 * will be completed using these custom variants instead of regular <stdlib.h>
 * ones.
 */
typedef void *(*ZSTD_allocFunction)(void *opaque, size_t size);
typedef void (*ZSTD_freeFunction)(void *opaque, void *address);
typedef struct {
  ZSTD_allocFunction customAlloc;
  ZSTD_freeFunction customFree;
  void *opaque;
} ZSTD_customMem;

/*! Custom memory allocation :
 *  These prototypes make it possible to pass your own allocation/free
 * functions. ZSTD_customMem is provided at creation time, using
 * ZSTD_create*_advanced() variants listed below. All allocation/free operations
 * will be completed using these custom variants instead of regular <stdlib.h>
 * ones.
 */
#if defined(__clang__) && __clang_major__ >= 5
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif
static
#ifdef __GNUC__
    __attribute__((__unused__))
#endif
    ZSTD_customMem const ZSTD_defaultCMem = {
        nullptr, nullptr,
        nullptr}; /**< this constant defers to stdlib's functions */
#if defined(__clang__) && __clang_major__ >= 5
#pragma clang diagnostic pop
#endif

// end from zstd.h

// from lib/common/allocations.h, adapted

#define ZSTD_malloc(s) malloc(s)
#define ZSTD_free(p) free((p))

DEVICE_INLINE void *ZSTD_calloc(size_t num, size_t size) {
  void *const ptr = ZSTD_malloc(num * size);
  ZSTD_memset(ptr, 0, num * size);
  return ptr;
}

DEVICE_INLINE void *ZSTD_customMalloc(size_t size, ZSTD_customMem customMem) {
  return ZSTD_malloc(size);
}

DEVICE_INLINE void *ZSTD_customCalloc(size_t size, ZSTD_customMem customMem) {
  // TODO(HIP/AMD): consider getting hipMM in here
  // if (customMem.customAlloc) {
  //     /* calloc implemented as malloc+memset;
  //      * not as efficient as calloc, but next best guess for custom malloc */
  //     void* const ptr = customMem.customAlloc(customMem.opaque, size);
  //     ZSTD_memset(ptr, 0, size);
  //     return ptr;
  // }
  return ZSTD_calloc(1, size);
}

DEVICE_INLINE void ZSTD_customFree(void *ptr, ZSTD_customMem customMem) {
  ZSTD_free(ptr);
}

// end from lib/common/allocations.h

// from lib/common/zstd_trace.h
#include <stddef.h>

/* weak symbol support
 * For now, enable conservatively:
 * - Only GNUC
 * - Only ELF
 * - Only x86-64, i386, aarch64 and risc-v.
 * Also, explicitly disable on platforms known not to work so they aren't
 * forgotten in the future.
 */
#if !defined(ZSTD_HAVE_WEAK_SYMBOLS) && defined(__GNUC__) &&                   \
    defined(__ELF__) &&                                                        \
    (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||            \
     defined(_M_IX86) || defined(__aarch64__) || defined(__riscv)) &&          \
    !defined(__APPLE__) && !defined(_WIN32) && !defined(__MINGW32__) &&        \
    !defined(__CYGWIN__) && !defined(_AIX)
#define ZSTD_HAVE_WEAK_SYMBOLS 1
#else
#define ZSTD_HAVE_WEAK_SYMBOLS 0
#endif
#if ZSTD_HAVE_WEAK_SYMBOLS
#define ZSTD_WEAK_ATTR __attribute__((__weak__))
#else
#define ZSTD_WEAK_ATTR
#endif

/* Only enable tracing when weak symbols are available. */
// #ifndef ZSTD_TRACE
// #define ZSTD_TRACE ZSTD_HAVE_WEAK_SYMBOLS
// #endif

// TODO(HIP/AMD): We disable tracing
#ifdef ZSTD_TRACE
#warning HIP/AMD ZSTD_TRACE option is currently not supported.
#undef ZSTD_TRACE // note: evaluates to 0
#endif

#if ZSTD_TRACE

struct ZSTD_CCtx_s;
struct ZSTD_DCtx_s;
struct ZSTD_CCtx_params_s;

typedef struct {
  /**
   * ZSTD_VERSION_NUMBER
   *
   * This is guaranteed to be the first member of ZSTD_trace.
   * Otherwise, this struct is not stable between versions. If
   * the version number does not match your expectation, you
   * should not interpret the rest of the struct.
   */
  unsigned version;
  /**
   * Non-zero if streaming (de)compression is used.
   */
  int streaming;
  /**
   * The dictionary ID.
   */
  unsigned dictionaryID;
  /**
   * Is the dictionary cold?
   * Only set on decompression.
   */
  int dictionaryIsCold;
  /**
   * The dictionary size or zero if no dictionary.
   */
  size_t dictionarySize;
  /**
   * The uncompressed size of the data.
   */
  size_t uncompressedSize;
  /**
   * The compressed size of the data.
   */
  size_t compressedSize;
  /**
   * The fully resolved CCtx parameters (NULL on decompression).
   */
  struct ZSTD_CCtx_params_s const *params;
  /**
   * The ZSTD_CCtx pointer (NULL on decompression).
   */
  struct ZSTD_CCtx_s const *cctx;
  /**
   * The ZSTD_DCtx pointer (NULL on compression).
   */
  struct ZSTD_DCtx_s const *dctx;
} ZSTD_Trace;

/**
 * A tracing context. It must be 0 when tracing is disabled.
 * Otherwise, any non-zero value returned by a tracing begin()
 * function is presented to any subsequent calls to end().
 *
 * Any non-zero value is treated as tracing is enabled and not
 * interpreted by the library.
 *
 * Two possible uses are:
 * * A timestamp for when the begin() function was called.
 * * A unique key identifying the (de)compression, like the
 *   address of the [dc]ctx pointer if you need to track
 *   more information than just a timestamp.
 */
typedef unsigned long long ZSTD_TraceCtx;

/**
 * Trace the beginning of a compression call.
 * @param cctx The dctx pointer for the compression.
 *             It can be used as a key to map begin() to end().
 * @returns Non-zero if tracing is enabled. The return value is
 *          passed to ZSTD_trace_compress_end().
 */
ZSTD_WEAK_ATTR ZSTD_TraceCtx
ZSTD_trace_compress_begin(struct ZSTD_CCtx_s const *cctx);

/**
 * Trace the end of a compression call.
 * @param ctx The return value of ZSTD_trace_compress_begin().
 * @param trace The zstd tracing info.
 */
ZSTD_WEAK_ATTR void ZSTD_trace_compress_end(ZSTD_TraceCtx ctx,
                                            ZSTD_Trace const *trace);

/**
 * Trace the beginning of a decompression call.
 * @param dctx The dctx pointer for the decompression.
 *             It can be used as a key to map begin() to end().
 * @returns Non-zero if tracing is enabled. The return value is
 *          passed to ZSTD_trace_compress_end().
 */
ZSTD_WEAK_ATTR ZSTD_TraceCtx
ZSTD_trace_decompress_begin(struct ZSTD_DCtx_s const *dctx);

/**
 * Trace the end of a decompression call.
 * @param ctx The return value of ZSTD_trace_decompress_begin().
 * @param trace The zstd tracing info.
 */
ZSTD_WEAK_ATTR void ZSTD_trace_decompress_end(ZSTD_TraceCtx ctx,
                                              ZSTD_Trace const *trace);

#endif /* ZSTD_TRACE */

} // namespace zstd
} // namespace hipcomp
