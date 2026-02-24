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

// NOTE: Derived from zstd/lib/common/xxhash.h

#pragma once

#include <cstdint>

#ifndef DEVICE_INLINE
#define DEVICE_INLINE __device__ inline
#endif

/*!
 * @def XXH_FORCE_ALIGN_CHECK
 * @brief If defined to non-zero, adds a special path for aligned inputs
 * (XXH32() and XXH64() only).
 *
 * This is an important performance trick for architectures without decent
 * unaligned memory access performance.
 *
 * It checks for input alignment, and when conditions are met, uses a "fast
 * path" employing direct 32-bit/64-bit reads, resulting in _dramatically
 * faster_ read speed.
 *
 * The check costs one initial branch per hash, which is generally negligible,
 * but not zero.
 *
 * Moreover, it's not useful to generate an additional code path if memory
 * access uses the same instruction for both aligned and unaligned
 * addresses (e.g. x86 and aarch64).
 *
 * In these cases, the alignment check can be removed by setting this macro to
 * 0. Then the code will always use unaligned memory access. Align check is
 * automatically disabled on x86, x64, ARM64, and some ARM chips which are
 * platforms known to offer good unaligned memory accesses performance.
 *
 * It is also disabled by default when @ref XXH_SIZE_OPT >= 1.
 *
 * This option does not affect XXH3 (only XXH32 and XXH64).
 */
#define XXH_FORCE_ALIGN_CHECK 0

/* ***   Endianness   *** */

/*!
 * @ingroup tuning
 * @def XXH_CPU_LITTLE_ENDIAN
 * @brief Whether the target is little endian.
 *
 * Defined to 1 if the target is little endian, or 0 if it is big endian.
 * It can be defined externally, for example on the compiler command line.
 *
 * If it is not defined,
 * a runtime check (which is usually constant folded) is used instead.
 *
 * @note
 *   This is not necessarily defined to an integer constant.
 *
 * @see XXH_isLittleEndian() for the runtime check.
 */
#ifndef XXH_CPU_LITTLE_ENDIAN
/*
 * Try to detect endianness automatically, to avoid the nonstandard behavior
 * in `XXH_isLittleEndian()`
 */
#if defined(_WIN32) /* Windows is always little endian */                      \
    || defined(__LITTLE_ENDIAN__) ||                                           \
    (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
#define XXH_CPU_LITTLE_ENDIAN 1
#elif defined(__BIG_ENDIAN__) ||                                               \
    (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#define XXH_CPU_LITTLE_ENDIAN 0
#else
/*!
 * @internal
 * @brief Runtime check for @ref XXH_CPU_LITTLE_ENDIAN.
 *
 * Most compilers will constant fold this.
 */
static int XXH_isLittleEndian(void) {
  /*
   * Portable and well-defined behavior.
   * Don't use static: it is detrimental to performance.
   */
  const union {
    xxh_u32 u;
    xxh_u8 c[4];
  } one = {1};
  return one.c[0];
}
#define XXH_CPU_LITTLE_ENDIAN XXH_isLittleEndian()
#endif
#endif

#if (XXH_DEBUGLEVEL >= 1)
#include <assert.h> /* note: can still be disabled with NDEBUG */
#define XXH_ASSERT(c) assert(c)
#else
#define XXH_ASSERT(c)
#endif

/*! @cond Doxygen ignores this part */
#ifdef __has_attribute
#define XXH_HAS_ATTRIBUTE(x) __has_attribute(x)
#else
#define XXH_HAS_ATTRIBUTE(x) 0
#endif

#ifdef __has_builtin
#define XXH_HAS_BUILTIN(x) __has_builtin(x)
#else
#define XXH_HAS_BUILTIN(x) 0
#endif

/*! @endcond */

/*! @cond Doxygen ignores this part */
/*
 * Define XXH_NOESCAPE for annotated pointers in public API.
 * https://clang.llvm.org/docs/AttributeReference.html#noescape
 * As of writing this, only supported by clang.
 */
#if XXH_HAS_ATTRIBUTE(noescape)
#define XXH_NOESCAPE __attribute__((noescape))
#else
#define XXH_NOESCAPE
#endif
/*! @endcond */

#if defined(__GNUC__)
#define XXH_CONSTF __attribute__((const))
#define XXH_PUREF __attribute__((pure))
#define XXH_MALLOCF __attribute__((malloc))
#else
#define XXH_CONSTF /* disable */
#define XXH_PUREF
#define XXH_MALLOCF
#endif

namespace hipcomp {
namespace zstd {

/* *************************************
 *  Basic Types
 ***************************************/
#include <cstdint>
typedef uint8_t xxh_u8;
typedef uint32_t XXH32_hash_t;
typedef uint64_t
    XXH64_hash_t; // TODO(HIP/AMD): Check if this is the correct datatype
typedef XXH32_hash_t xxh_u32;
typedef XXH64_hash_t xxh_u64;

/*!
 * @brief Exit code for the streaming API.
 */
typedef enum {
  XXH_OK = 0, /*!< OK */
  XXH_ERROR   /*!< Error */
} XXH_errorcode;

typedef struct XXH64_state_s XXH64_state_t;

struct XXH64_state_s {
  XXH64_hash_t total_len;
  XXH64_hash_t v[4];
  XXH64_hash_t mem64[4];
  XXH32_hash_t memsize;
  XXH32_hash_t reserved32;
  XXH64_hash_t reserved64;
};

/*******   xxh64   *******/
/*!
 * @}
 * @defgroup XXH64_impl XXH64 implementation
 * @ingroup impl
 *
 * Details on the XXH64 implementation.
 * @{
 */
/* #define rather that static const, to be used as initializers */
#define XXH_PRIME64_1                                                                         \
  0x9E3779B185EBCA87ULL /*!<                                                                  \
                           0b1001111000110111011110011011000110000101111010111100101010000111 \
                         */
#define XXH_PRIME64_2                                                                         \
  0xC2B2AE3D27D4EB4FULL /*!<                                                                  \
                           0b1100001010110010101011100011110100100111110101001110101101001111 \
                         */
#define XXH_PRIME64_3                                                                         \
  0x165667B19E3779F9ULL /*!<                                                                  \
                           0b0001011001010110011001111011000110011110001101110111100111111001 \
                         */
#define XXH_PRIME64_4                                                                         \
  0x85EBCA77C2B2AE63ULL /*!<                                                                  \
                           0b1000010111101011110010100111011111000010101100101010111001100011 \
                         */
#define XXH_PRIME64_5                                                                         \
  0x27D4EB2F165667C5ULL /*!<                                                                  \
                           0b0010011111010100111010110010111100010110010101100110011111000101 \
                         */

#if !defined(NO_CLANG_BUILTIN) && XXH_HAS_BUILTIN(__builtin_rotateleft32) &&   \
    XXH_HAS_BUILTIN(__builtin_rotateleft64)
#define XXH_rotl32 __builtin_rotateleft32
#define XXH_rotl64 __builtin_rotateleft64
/* Note: although _rotl exists for minGW (GCC under windows), performance seems
 * poor */
#elif defined(_MSC_VER)
#define XXH_rotl32(x, r) _rotl(x, r)
#define XXH_rotl64(x, r) _rotl64(x, r)
#else
#define XXH_rotl32(x, r) (((x) << (r)) | ((x) >> (32 - (r))))
#define XXH_rotl64(x, r) (((x) << (r)) | ((x) >> (64 - (r))))
#endif

/*!
 * @internal
 * @fn xxh_u32 XXH_swap32(xxh_u32 x)
 * @brief A 32-bit byteswap.
 *
 * @param x The 32-bit integer to byteswap.
 * @return @p x, byteswapped.
 */
#if defined(_MSC_VER) /* Visual Studio */
#define XXH_swap32 _byteswap_ulong
#elif XXH_GCC_VERSION >= 403
#define XXH_swap32 __builtin_bswap32
#else
static xxh_u32 XXH_swap32(xxh_u32 x) {
  return ((x << 24) & 0xff000000) | ((x << 8) & 0x00ff0000) |
         ((x >> 8) & 0x0000ff00) | ((x >> 24) & 0x000000ff);
}
#endif

/*!
 * @internal
 * @fn xxh_u32 XXH_swap32(xxh_u32 x)
 * @brief A 32-bit byteswap.
 *
 * @param x The 32-bit integer to byteswap.
 * @return @p x, byteswapped.
 */
#if defined(_MSC_VER) /* Visual Studio */
#define XXH_swap32 _byteswap_ulong
#elif XXH_GCC_VERSION >= 403
#define XXH_swap32 __builtin_bswap32
#else
DEVICE_INLINE xxh_u32 XXH_swap32(xxh_u32 x) {
  return ((x << 24) & 0xff000000) | ((x << 8) & 0x00ff0000) |
         ((x >> 8) & 0x0000ff00) | ((x >> 24) & 0x000000ff);
}
#endif

#if defined(_MSC_VER) /* Visual Studio */
#define XXH_swap64 _byteswap_uint64
#elif XXH_GCC_VERSION >= 403
#define XXH_swap64 __builtin_bswap64
#else
DEVICE_INLINE xxh_u64 XXH_swap64(xxh_u64 x) {
  return ((x << 56) & 0xff00000000000000ULL) |
         ((x << 40) & 0x00ff000000000000ULL) |
         ((x << 24) & 0x0000ff0000000000ULL) |
         ((x << 8) & 0x000000ff00000000ULL) |
         ((x >> 8) & 0x00000000ff000000ULL) |
         ((x >> 24) & 0x0000000000ff0000ULL) |
         ((x >> 40) & 0x000000000000ff00ULL) |
         ((x >> 56) & 0x00000000000000ffULL);
}
#endif

/*!
 * @internal
 * @brief Modify this function to use a different routine than memcpy().
 */
DEVICE_INLINE void *XXH_memcpy(void *dest, const void *src, size_t size) {
  return memcpy(dest, src, size);
}

/*!
 * @internal
 * @brief Enum to indicate whether a pointer is aligned.
 */
typedef enum {
  XXH_aligned,  /*!< Aligned */
  XXH_unaligned /*!< Possibly unaligned */
} XXH_alignment;

/*!
 * @internal
 * @fn xxh_u32 XXH_readLE32_align(const void* ptr, XXH_alignment align)
 * @brief Like @ref XXH_readLE32(), but has an option for aligned reads.
 *
 * Affected by @ref XXH_FORCE_MEMORY_ACCESS.
 * Note that when @ref XXH_FORCE_ALIGN_CHECK == 0, the @p align parameter is
 * always @ref XXH_alignment::XXH_unaligned.
 *
 * @param ptr The pointer to read from.
 * @param align Whether @p ptr is aligned.
 * @pre
 *   If @p align == @ref XXH_alignment::XXH_aligned, @p ptr must be 4 byte
 *   aligned.
 * @return The 32-bit little endian integer from the bytes at @p ptr.
 */

#if (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS == 3))
/*
 * Manual byteshift. Best for old compilers which don't inline memcpy.
 * We actually directly use XXH_readLE32 and XXH_readBE32.
 */
#elif (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS == 2))

/*
 * Force direct memory access. Only works on CPU which support unaligned memory
 * access in hardware.
 */
static xxh_u32 XXH_read32(const void *memPtr) {
  return *(const xxh_u32 *)memPtr;
}

#elif (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS == 1))

/*
 * __attribute__((aligned(1))) is supported by gcc and clang. Originally the
 * documentation claimed that it only increased the alignment, but actually it
 * can decrease it on gcc, clang, and icc:
 * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=69502,
 * https://gcc.godbolt.org/z/xYez1j67Y.
 */
#ifdef XXH_OLD_NAMES
typedef union {
  xxh_u32 u32;
} __attribute__((packed)) unalign;
#endif
DEVICE_INLINE xxh_u32 XXH_read32(const void *ptr) {
  typedef __attribute__((aligned(1))) xxh_u32 xxh_unalign32;
  return *((const xxh_unalign32 *)ptr);
}

#else

/*
 * Portable and safe solution. Generally efficient.
 * see:
 * https://fastcompression.blogspot.com/2015/08/accessing-unaligned-memory.html
 */
DEVICE_INLINE xxh_u32 XXH_read32(const void *memPtr) {
  xxh_u32 val;
  XXH_memcpy(&val, memPtr, sizeof(val));
  return val;
}

#endif /* XXH_FORCE_DIRECT_MEMORY_ACCESS */

#if (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS == 3))
/*
 * Manual byteshift. Best for old compilers which don't inline memcpy.
 * We actually directly use XXH_readLE64 and XXH_readBE64.
 */
#elif (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS == 2))

/* Force direct memory access. Only works on CPU which support unaligned memory
 * access in hardware */
DEVICE_INLINE xxh_u64 XXH_read64(const void *memPtr) {
  return *(const xxh_u64 *)memPtr;
}

#elif (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS == 1))

/*
 * __attribute__((aligned(1))) is supported by gcc and clang. Originally the
 * documentation claimed that it only increased the alignment, but actually it
 * can decrease it on gcc, clang, and icc:
 * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=69502,
 * https://gcc.godbolt.org/z/xYez1j67Y.
 */
#ifdef XXH_OLD_NAMES
typedef union {
  xxh_u32 u32;
  xxh_u64 u64;
} __attribute__((packed)) unalign64;
#endif
DEVICE_INLINE xxh_u64 XXH_read64(const void *ptr) {
  typedef __attribute__((aligned(1))) xxh_u64 xxh_unalign64;
  return *((const xxh_unalign64 *)ptr);
}

#else

/*
 * Portable and safe solution. Generally efficient.
 * see:
 * https://fastcompression.blogspot.com/2015/08/accessing-unaligned-memory.html
 */
DEVICE_INLINE xxh_u64 XXH_read64(const void *memPtr) {
  xxh_u64 val;
  XXH_memcpy(&val, memPtr, sizeof(val));
  return val;
}

#endif /* XXH_FORCE_DIRECT_MEMORY_ACCESS */

/*! @copydoc XXH32_round */
DEVICE_INLINE xxh_u64 XXH64_round(xxh_u64 acc, xxh_u64 input) {
  acc += input * XXH_PRIME64_2;
  acc = XXH_rotl64(acc, 31);
  acc *= XXH_PRIME64_1;
  // NOTE(HIP/AMD): removed AVX512 specific path
  return acc;
}

DEVICE_INLINE xxh_u64 XXH64_mergeRound(xxh_u64 acc, xxh_u64 val) {
  val = XXH64_round(0, val);
  acc ^= val;
  acc = acc * XXH_PRIME64_1 + XXH_PRIME64_4;
  return acc;
}

/*
 * XXH_FORCE_MEMORY_ACCESS==3 is an endian-independent byteshift load.
 *
 * This is ideal for older compilers which don't inline memcpy.
 */
#if (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS == 3))

DEVICE_INLINE xxh_u32 XXH_readLE32(const void *memPtr) {
  const xxh_u8 *bytePtr = (const xxh_u8 *)memPtr;
  return bytePtr[0] | ((xxh_u32)bytePtr[1] << 8) | ((xxh_u32)bytePtr[2] << 16) |
         ((xxh_u32)bytePtr[3] << 24);
}

#else
DEVICE_INLINE xxh_u32 XXH_readLE32(const void *ptr) {
  return XXH_CPU_LITTLE_ENDIAN ? XXH_read32(ptr) : XXH_swap32(XXH_read32(ptr));
}
#endif

DEVICE_INLINE xxh_u32 XXH_readLE32_align(const void *ptr, XXH_alignment align) {
  if (align == XXH_unaligned) {
    return XXH_readLE32(ptr);
  } else {
    return XXH_CPU_LITTLE_ENDIAN ? *(const xxh_u32 *)ptr
                                 : XXH_swap32(*(const xxh_u32 *)ptr);
  }
}

/* XXH_FORCE_MEMORY_ACCESS==3 is an endian-independent byteshift load. */
#if (defined(XXH_FORCE_MEMORY_ACCESS) && (XXH_FORCE_MEMORY_ACCESS == 3))

DEVICE_INLINE xxh_u64 XXH_readLE64(const void *memPtr) {
  const xxh_u8 *bytePtr = (const xxh_u8 *)memPtr;
  return bytePtr[0] | ((xxh_u64)bytePtr[1] << 8) | ((xxh_u64)bytePtr[2] << 16) |
         ((xxh_u64)bytePtr[3] << 24) | ((xxh_u64)bytePtr[4] << 32) |
         ((xxh_u64)bytePtr[5] << 40) | ((xxh_u64)bytePtr[6] << 48) |
         ((xxh_u64)bytePtr[7] << 56);
}

#else
DEVICE_INLINE xxh_u64 XXH_readLE64(const void *ptr) {
  return XXH_CPU_LITTLE_ENDIAN ? XXH_read64(ptr) : XXH_swap64(XXH_read64(ptr));
}
#endif

DEVICE_INLINE xxh_u64 XXH_readLE64_align(const void *ptr, XXH_alignment align) {
  if (align == XXH_unaligned)
    return XXH_readLE64(ptr);
  else
    return XXH_CPU_LITTLE_ENDIAN ? *(const xxh_u64 *)ptr
                                 : XXH_swap64(*(const xxh_u64 *)ptr);
}

#define XXH_get32bits(p) XXH_readLE32_align(p, align)
#define XXH_get64bits(p) XXH_readLE64_align(p, align)

/*! @copydoc XXH32_avalanche */
DEVICE_INLINE xxh_u64 XXH64_avalanche(xxh_u64 hash) {
  hash ^= hash >> 33;
  hash *= XXH_PRIME64_2;
  hash ^= hash >> 29;
  hash *= XXH_PRIME64_3;
  hash ^= hash >> 32;
  return hash;
}

/*!
 * @internal
 * @brief Processes the last 0-31 bytes of @p ptr.
 *
 * There may be up to 31 bytes remaining to consume from the input.
 * This final stage will digest them to ensure that all input bytes are present
 * in the final mix.
 *
 * @param hash The hash to finalize.
 * @param ptr The pointer to the remaining input.
 * @param len The remaining length, modulo 32.
 * @param align Whether @p ptr is aligned.
 * @return The finalized hash
 * @see XXH32_finalize().
 */
DEVICE_INLINE xxh_u64 XXH64_finalize(xxh_u64 hash, const xxh_u8 *ptr,
                                     size_t len, XXH_alignment align) {
  if (ptr == NULL)
    XXH_ASSERT(len == 0);
  len &= 31;
  while (len >= 8) {
    xxh_u64 const k1 = XXH64_round(0, XXH_get64bits(ptr));
    ptr += 8;
    hash ^= k1;
    hash = XXH_rotl64(hash, 27) * XXH_PRIME64_1 + XXH_PRIME64_4;
    len -= 8;
  }
  if (len >= 4) {
    hash ^= (xxh_u64)(XXH_get32bits(ptr)) * XXH_PRIME64_1;
    ptr += 4;
    hash = XXH_rotl64(hash, 23) * XXH_PRIME64_2 + XXH_PRIME64_3;
    len -= 4;
  }
  while (len > 0) {
    hash ^= (*ptr++) * XXH_PRIME64_5;
    hash = XXH_rotl64(hash, 11) * XXH_PRIME64_1;
    --len;
  }
  return XXH64_avalanche(hash);
}

/*! @ingroup XXH64_family */
DEVICE_INLINE XXH_errorcode XXH64_update(XXH_NOESCAPE XXH64_state_t *state,
                                         XXH_NOESCAPE const void *input,
                                         size_t len) {
  if (input == NULL) {
    XXH_ASSERT(len == 0);
    return XXH_OK;
  }

  {
    const xxh_u8 *p = (const xxh_u8 *)input;
    const xxh_u8 *const bEnd = p + len;

    state->total_len += len;

    if (state->memsize + len < 32) { /* fill in tmp buffer */
      XXH_memcpy(((xxh_u8 *)state->mem64) + state->memsize, input, len);
      state->memsize += (xxh_u32)len;
      return XXH_OK;
    }

    if (state->memsize) { /* tmp buffer is full */
      XXH_memcpy(((xxh_u8 *)state->mem64) + state->memsize, input,
                 32 - state->memsize);
      state->v[0] = XXH64_round(state->v[0], XXH_readLE64(state->mem64 + 0));
      state->v[1] = XXH64_round(state->v[1], XXH_readLE64(state->mem64 + 1));
      state->v[2] = XXH64_round(state->v[2], XXH_readLE64(state->mem64 + 2));
      state->v[3] = XXH64_round(state->v[3], XXH_readLE64(state->mem64 + 3));
      p += 32 - state->memsize;
      state->memsize = 0;
    }

    if (p + 32 <= bEnd) {
      const xxh_u8 *const limit = bEnd - 32;

      do {
        state->v[0] = XXH64_round(state->v[0], XXH_readLE64(p));
        p += 8;
        state->v[1] = XXH64_round(state->v[1], XXH_readLE64(p));
        p += 8;
        state->v[2] = XXH64_round(state->v[2], XXH_readLE64(p));
        p += 8;
        state->v[3] = XXH64_round(state->v[3], XXH_readLE64(p));
        p += 8;
      } while (p <= limit);
    }

    if (p < bEnd) {
      XXH_memcpy(state->mem64, p, (size_t)(bEnd - p));
      state->memsize = (unsigned)(bEnd - p);
    }
  }

  return XXH_OK;
}

/*! @ingroup XXH64_family */
DEVICE_INLINE XXH_errorcode XXH64_reset(XXH_NOESCAPE XXH64_state_t *statePtr,
                                        XXH64_hash_t seed) {
  XXH_ASSERT(statePtr != NULL);
  memset(statePtr, 0, sizeof(*statePtr));
  statePtr->v[0] = seed + XXH_PRIME64_1 + XXH_PRIME64_2;
  statePtr->v[1] = seed + XXH_PRIME64_2;
  statePtr->v[2] = seed + 0;
  statePtr->v[3] = seed - XXH_PRIME64_1;
  return XXH_OK;
}

/*! @ingroup XXH64_family */
DEVICE_INLINE XXH64_hash_t
XXH64_digest(XXH_NOESCAPE const XXH64_state_t *state) {
  xxh_u64 h64;

  if (state->total_len >= 32) {
    h64 = XXH_rotl64(state->v[0], 1) + XXH_rotl64(state->v[1], 7) +
          XXH_rotl64(state->v[2], 12) + XXH_rotl64(state->v[3], 18);
    h64 = XXH64_mergeRound(h64, state->v[0]);
    h64 = XXH64_mergeRound(h64, state->v[1]);
    h64 = XXH64_mergeRound(h64, state->v[2]);
    h64 = XXH64_mergeRound(h64, state->v[3]);
  } else {
    h64 = state->v[2] /*seed*/ + XXH_PRIME64_5;
  }

  h64 += (xxh_u64)state->total_len;

  return XXH64_finalize(h64, (const xxh_u8 *)state->mem64,
                        (size_t)state->total_len, XXH_aligned);
}

/*!
 * @internal
 * @brief The implementation for @ref XXH64().
 *
 * @param input , len , seed Directly passed from @ref XXH64().
 * @param align Whether @p input is aligned.
 * @return The calculated hash.
 */
DEVICE_INLINE XXH_PUREF xxh_u64 XXH64_endian_align(const xxh_u8 *input,
                                                   size_t len, xxh_u64 seed,
                                                   XXH_alignment align) {
  xxh_u64 h64;
  if (input == NULL)
    XXH_ASSERT(len == 0);

  if (len >= 32) {
    const xxh_u8 *const bEnd = input + len;
    const xxh_u8 *const limit = bEnd - 31;
    xxh_u64 v1 = seed + XXH_PRIME64_1 + XXH_PRIME64_2;
    xxh_u64 v2 = seed + XXH_PRIME64_2;
    xxh_u64 v3 = seed + 0;
    xxh_u64 v4 = seed - XXH_PRIME64_1;

    do {
      v1 = XXH64_round(v1, XXH_get64bits(input));
      input += 8;
      v2 = XXH64_round(v2, XXH_get64bits(input));
      input += 8;
      v3 = XXH64_round(v3, XXH_get64bits(input));
      input += 8;
      v4 = XXH64_round(v4, XXH_get64bits(input));
      input += 8;
    } while (input < limit);

    h64 = XXH_rotl64(v1, 1) + XXH_rotl64(v2, 7) + XXH_rotl64(v3, 12) +
          XXH_rotl64(v4, 18);
    h64 = XXH64_mergeRound(h64, v1);
    h64 = XXH64_mergeRound(h64, v2);
    h64 = XXH64_mergeRound(h64, v3);
    h64 = XXH64_mergeRound(h64, v4);

  } else {
    h64 = seed + XXH_PRIME64_5;
  }

  h64 += (xxh_u64)len;

  return XXH64_finalize(h64, input, len, align);
}

/*!
 * @brief Calculates the 64-bit hash of @p input using xxHash64.
 *
 * @param input The block of data to be hashed, at least @p length bytes in
 * size.
 * @param len The length of @p input, in bytes.
 * @param seed The 64-bit seed to alter the hash's output predictably.
 *
 * @pre
 *   The memory between @p input and @p input + @p length must be valid,
 *   readable, contiguous memory. However, if @p length is `0`, @p input may be
 *   `NULL`. In C++, this also must be *TriviallyCopyable*.
 *
 * @return The calculated 64-bit xxHash64 value.
 *
 * @see @ref single_shot_example "Single Shot Example" for an example.
 */
DEVICE_INLINE XXH64_hash_t XXH64(XXH_NOESCAPE const void *input, size_t len,
                                 XXH64_hash_t seed) {
#if !defined(XXH_NO_STREAM) && XXH_SIZE_OPT >= 2
  /* Simple version, good for code maintenance, but unfortunately slow for small
   * inputs */
  XXH64_state_t state;
  XXH64_reset(&state, seed);
  XXH64_update(&state, (const xxh_u8 *)input, len);
  return XXH64_digest(&state);
#else
  if (XXH_FORCE_ALIGN_CHECK) {
    if ((((size_t)input) & 7) ==
        0) { /* Input is aligned, let's leverage the speed advantage */
      return XXH64_endian_align((const xxh_u8 *)input, len, seed, XXH_aligned);
    }
  }

  return XXH64_endian_align((const xxh_u8 *)input, len, seed, XXH_unaligned);
#endif
}

} // namespace zstd
} // namespace hipcomp
