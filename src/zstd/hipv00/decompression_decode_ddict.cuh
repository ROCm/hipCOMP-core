/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

/* Derived from zstd_ddict.c :
 * concentrates all logic that needs to know the internals of ZSTD_DDict object
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

#include "common.cuh"
#include "decompression_decode_block.cuh"
#include "decompression_decode_fse.cuh"
#include "decompression_decode_huf.cuh"
#include "decompression_state.cuh"

namespace hipcomp {
namespace zstd {

// from zstd.h

typedef enum {
  ZSTD_dct_auto = 0,       /* dictionary is "full" when starting with
                              ZSTD_MAGIC_DICTIONARY, otherwise it is "rawContent" */
  ZSTD_dct_rawContent = 1, /* ensures dictionary is always loaded as rawContent,
                              even if it starts with ZSTD_MAGIC_DICTIONARY */
  ZSTD_dct_fullDict =
      2 /* refuses to load a dictionary if it does not respect Zstandard's
           specification, starting with ZSTD_MAGIC_DICTIONARY */
} ZSTD_dictContentType_e;

typedef enum {
  ZSTD_dlm_byCopy = 0, /**< Copy dictionary content internally */
  ZSTD_dlm_byRef = 1 /**< Reference dictionary content -- the dictionary buffer
                        must outlive its users. */
} ZSTD_dictLoadMethod_e;

// end from zstd.h

// from zstd_ddict.h|c

/*-*******************************************************
 *  Types
 *********************************************************/
struct ZSTD_DDict_s {
  void *dictBuffer;
  const void *dictContent;
  size_t dictSize;
  ZSTD_entropyDTables_t entropy;
  U32 dictID;
  U32 entropyPresent;
  ZSTD_customMem cMem;
}; /* typedef'd to ZSTD_DDict within "zstd.h" */

DEVICE_INLINE const void *ZSTD_DDict_dictContent(const ZSTD_DDict *ddict) {
  assert(ddict != NULL);
  return ddict->dictContent;
}

DEVICE_INLINE size_t ZSTD_DDict_dictSize(const ZSTD_DDict *ddict) {
  assert(ddict != NULL);
  return ddict->dictSize;
}

DEVICE_INLINE void ZSTD_copyDDictParameters(ZSTD_DCtx *dctx,
                                            const ZSTD_DDict *ddict) {
  DEBUGLOG(4, "ZSTD_copyDDictParameters");
  assert(dctx != NULL);
  assert(ddict != NULL);
  dctx->dictID = ddict->dictID;
  dctx->prefixStart = ddict->dictContent;
  dctx->virtualStart = ddict->dictContent;
  dctx->dictEnd = (const BYTE *)ddict->dictContent + ddict->dictSize;
  dctx->previousDstEnd = dctx->dictEnd;
  // #ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
  //     dctx->dictContentBeginForFuzzing = dctx->prefixStart;
  //     dctx->dictContentEndForFuzzing = dctx->previousDstEnd;
  // #endif
  if (ddict->entropyPresent) {
    dctx->litEntropy = 1;
    dctx->fseEntropy = 1;
    dctx->LLTptr = ddict->entropy.LLTable;
    dctx->MLTptr = ddict->entropy.MLTable;
    dctx->OFTptr = ddict->entropy.OFTable;
    dctx->HUFptr = ddict->entropy.hufTable;
    dctx->entropy.rep[0] = ddict->entropy.rep[0];
    dctx->entropy.rep[1] = ddict->entropy.rep[1];
    dctx->entropy.rep[2] = ddict->entropy.rep[2];
  } else {
    dctx->litEntropy = 0;
    dctx->fseEntropy = 0;
  }
}

//* from zstd_decompress.c

/*! ZSTD_loadDEntropy() :
 *  dict : must point at beginning of a valid zstd dictionary.
 * @return : size of dictionary header (size of magic number + dict ID + entropy
 * tables) */
DEVICE_INLINE size_t ZSTD_loadDEntropy(ZSTD_entropyDTables_t *entropy,
                                       const void *const dict,
                                       size_t const dictSize) {
  const BYTE *dictPtr = (const BYTE *)dict;
  const BYTE *const dictEnd = dictPtr + dictSize;

  RETURN_ERROR_IF(dictSize <= 8, dictionary_corrupted, "dict is too small");
  assert(MEM_readLE32(dict) == ZSTD_MAGIC_DICTIONARY); /* dict must be valid */
  dictPtr += 8; /* skip header = magic + dictID */

  ZSTD_STATIC_ASSERT(offsetof(ZSTD_entropyDTables_t, OFTable) ==
                     offsetof(ZSTD_entropyDTables_t, LLTable) +
                         sizeof(entropy->LLTable));
  ZSTD_STATIC_ASSERT(offsetof(ZSTD_entropyDTables_t, MLTable) ==
                     offsetof(ZSTD_entropyDTables_t, OFTable) +
                         sizeof(entropy->OFTable));
  ZSTD_STATIC_ASSERT(sizeof(entropy->LLTable) + sizeof(entropy->OFTable) +
                         sizeof(entropy->MLTable) >=
                     HUF_DECOMPRESS_WORKSPACE_SIZE);
  {
    void *const workspace =
        &entropy->LLTable; /* use fse tables as temporary workspace; implies fse
                              tables are grouped together */
    size_t const workspaceSize = sizeof(entropy->LLTable) +
                                 sizeof(entropy->OFTable) +
                                 sizeof(entropy->MLTable);
#ifdef HUF_FORCE_DECOMPRESS_X1
    /* in minimal huffman, we always use X1 variants */
    size_t const hSize =
        HUF_readDTableX1_wksp(entropy->hufTable, dictPtr, dictEnd - dictPtr,
                              workspace, workspaceSize, /* flags */ 0);
#else
    size_t const hSize = HUF_readDTableX2_wksp(
        entropy->hufTable, dictPtr, (size_t)(dictEnd - dictPtr), workspace,
        workspaceSize, /* flags */ 0);
#endif
    RETURN_ERROR_IF(HUF_isError(hSize), dictionary_corrupted, "");
    dictPtr += hSize;
  }

  {
    short offcodeNCount[MaxOff + 1];
    unsigned offcodeMaxValue = MaxOff, offcodeLog;
    size_t const offcodeHeaderSize =
        FSE_readNCount(offcodeNCount, &offcodeMaxValue, &offcodeLog, dictPtr,
                       (size_t)(dictEnd - dictPtr));
    RETURN_ERROR_IF(FSE_isError(offcodeHeaderSize), dictionary_corrupted, "");
    RETURN_ERROR_IF(offcodeMaxValue > MaxOff, dictionary_corrupted, "");
    RETURN_ERROR_IF(offcodeLog > OffFSELog, dictionary_corrupted, "");
    ZSTD_buildFSETable(entropy->OFTable, offcodeNCount, offcodeMaxValue,
                       OF_base, OF_bits, offcodeLog, entropy->workspace,
                       sizeof(entropy->workspace),
                       /* bmi2 */ 0);
    dictPtr += offcodeHeaderSize;
  }

  {
    short matchlengthNCount[MaxML + 1];
    unsigned matchlengthMaxValue = MaxML, matchlengthLog;
    size_t const matchlengthHeaderSize =
        FSE_readNCount(matchlengthNCount, &matchlengthMaxValue, &matchlengthLog,
                       dictPtr, (size_t)(dictEnd - dictPtr));
    RETURN_ERROR_IF(FSE_isError(matchlengthHeaderSize), dictionary_corrupted,
                    "");
    RETURN_ERROR_IF(matchlengthMaxValue > MaxML, dictionary_corrupted, "");
    RETURN_ERROR_IF(matchlengthLog > MLFSELog, dictionary_corrupted, "");
    ZSTD_buildFSETable(entropy->MLTable, matchlengthNCount, matchlengthMaxValue,
                       ML_base, ML_bits, matchlengthLog, entropy->workspace,
                       sizeof(entropy->workspace),
                       /* bmi2 */ 0);
    dictPtr += matchlengthHeaderSize;
  }

  {
    short litlengthNCount[MaxLL + 1];
    unsigned litlengthMaxValue = MaxLL, litlengthLog;
    size_t const litlengthHeaderSize =
        FSE_readNCount(litlengthNCount, &litlengthMaxValue, &litlengthLog,
                       dictPtr, (size_t)(dictEnd - dictPtr));
    RETURN_ERROR_IF(FSE_isError(litlengthHeaderSize), dictionary_corrupted, "");
    RETURN_ERROR_IF(litlengthMaxValue > MaxLL, dictionary_corrupted, "");
    RETURN_ERROR_IF(litlengthLog > LLFSELog, dictionary_corrupted, "");
    ZSTD_buildFSETable(entropy->LLTable, litlengthNCount, litlengthMaxValue,
                       LL_base, LL_bits, litlengthLog, entropy->workspace,
                       sizeof(entropy->workspace),
                       /* bmi2 */ 0);
    dictPtr += litlengthHeaderSize;
  }

  RETURN_ERROR_IF(dictPtr + 12 > dictEnd, dictionary_corrupted, "");
  {
    int i;
    size_t const dictContentSize = (size_t)(dictEnd - (dictPtr + 12));
    for (i = 0; i < 3; i++) {
      U32 const rep = MEM_readLE32(dictPtr);
      dictPtr += 4;
      RETURN_ERROR_IF(rep == 0 || rep > dictContentSize, dictionary_corrupted,
                      "");
      entropy->rep[i] = rep;
    }
  }

  return (size_t)(dictPtr - (const BYTE *)dict);
}

//* end from zstd_decompress_internal.c

DEVICE_INLINE static size_t
ZSTD_loadEntropy_intoDDict(ZSTD_DDict *ddict,
                           ZSTD_dictContentType_e dictContentType) {
  ddict->dictID = 0;
  ddict->entropyPresent = 0;
  if (dictContentType == ZSTD_dct_rawContent)
    return 0;

  if (ddict->dictSize < 8) {
    if (dictContentType == ZSTD_dct_fullDict)
      return ERROR(
          dictionary_corrupted); /* only accept specified dictionaries */
    return 0;                    /* pure content mode */
  }
  {
    U32 const magic = MEM_readLE32(ddict->dictContent);
    if (magic != ZSTD_MAGIC_DICTIONARY) {
      if (dictContentType == ZSTD_dct_fullDict)
        return ERROR(
            dictionary_corrupted); /* only accept specified dictionaries */
      return 0;                    /* pure content mode */
    }
  }
  ddict->dictID =
      MEM_readLE32((const char *)ddict->dictContent + ZSTD_FRAMEIDSIZE);

  /* load entropy tables */
  RETURN_ERROR_IF(ZSTD_isError(ZSTD_loadDEntropy(
                      &ddict->entropy, ddict->dictContent, ddict->dictSize)),
                  dictionary_corrupted, "");
  ddict->entropyPresent = 1;
  return 0;
}

DEVICE_INLINE static size_t
ZSTD_initDDict_internal(ZSTD_DDict *ddict, const void *dict, size_t dictSize,
                        ZSTD_dictLoadMethod_e dictLoadMethod,
                        ZSTD_dictContentType_e dictContentType) {
  if ((dictLoadMethod == ZSTD_dlm_byRef) || (!dict) || (!dictSize)) {
    ddict->dictBuffer = NULL;
    ddict->dictContent = dict;
    if (!dict)
      dictSize = 0;
  } else {
    void *const internalBuffer = ZSTD_customMalloc(dictSize, ddict->cMem);
    ddict->dictBuffer = internalBuffer;
    ddict->dictContent = internalBuffer;
    if (!internalBuffer)
      return ERROR(memory_allocation);
    ZSTD_memcpy(internalBuffer, dict, dictSize);
  }
  ddict->dictSize = dictSize;
  ddict->entropy.hufTable[0] =
      (HUF_DTable)((ZSTD_HUFFDTABLE_CAPACITY_LOG) *
                   0x1000001); /* cover both little and big endian */

  /* parse dictionary content */
  FORWARD_IF_ERROR(ZSTD_loadEntropy_intoDDict(ddict, dictContentType), "");

  return 0;
}

DEVICE_INLINE size_t ZSTD_freeDDict(ZSTD_DDict *ddict) {
  if (ddict == NULL)
    return 0; /* support free on NULL */
  {
    ZSTD_customMem const cMem = ddict->cMem;
    ZSTD_customFree(ddict->dictBuffer, cMem);
    ZSTD_customFree(ddict, cMem);
    return 0;
  }
}

DEVICE_INLINE ZSTD_DDict *ZSTD_createDDict_advanced(
    const void *dict, size_t dictSize, ZSTD_dictLoadMethod_e dictLoadMethod,
    ZSTD_dictContentType_e dictContentType, ZSTD_customMem customMem) {
  if ((!customMem.customAlloc) ^ (!customMem.customFree))
    return NULL;

  {
    ZSTD_DDict *const ddict =
        (ZSTD_DDict *)ZSTD_customMalloc(sizeof(ZSTD_DDict), customMem);
    if (ddict == NULL)
      return NULL;
    ddict->cMem = customMem;
    {
      size_t const initResult = ZSTD_initDDict_internal(
          ddict, dict, dictSize, dictLoadMethod, dictContentType);
      if (ZSTD_isError(initResult)) {
        ZSTD_freeDDict(ddict);
        return NULL;
      }
    }
    return ddict;
  }
}

/*! ZSTD_createDDict() :
 *   Create a digested dictionary, to start decompression without startup delay.
 *   `dict` content is copied inside DDict.
 *   Consequently, `dict` can be released after `ZSTD_DDict` creation */
DEVICE_INLINE ZSTD_DDict *ZSTD_createDDict(const void *dict, size_t dictSize) {
  ZSTD_customMem const allocator = {NULL, NULL, NULL};
  return ZSTD_createDDict_advanced(dict, dictSize, ZSTD_dlm_byCopy,
                                   ZSTD_dct_auto, allocator);
}

/*! ZSTD_createDDict_byReference() :
 *  Create a digested dictionary, to start decompression without startup delay.
 *  Dictionary content is simply referenced, it will be accessed during
 * decompression. Warning : dictBuffer must outlive DDict (DDict must be freed
 * before dictBuffer) */
DEVICE_INLINE ZSTD_DDict *ZSTD_createDDict_byReference(const void *dictBuffer,
                                                       size_t dictSize) {
  ZSTD_customMem const allocator = {NULL, NULL, NULL};
  return ZSTD_createDDict_advanced(dictBuffer, dictSize, ZSTD_dlm_byRef,
                                   ZSTD_dct_auto, allocator);
}

DEVICE_INLINE const ZSTD_DDict *
ZSTD_initStaticDDict(void *sBuffer, size_t sBufferSize, const void *dict,
                     size_t dictSize, ZSTD_dictLoadMethod_e dictLoadMethod,
                     ZSTD_dictContentType_e dictContentType) {
  size_t const neededSpace =
      sizeof(ZSTD_DDict) + (dictLoadMethod == ZSTD_dlm_byRef ? 0 : dictSize);
  ZSTD_DDict *const ddict = (ZSTD_DDict *)sBuffer;
  assert(sBuffer != NULL);
  assert(dict != NULL);
  if ((size_t)sBuffer & 7)
    return NULL; /* 8-aligned */
  if (sBufferSize < neededSpace)
    return NULL;
  if (dictLoadMethod == ZSTD_dlm_byCopy) {
    ZSTD_memcpy(ddict + 1, dict, dictSize); /* local copy */
    dict = ddict + 1;
  }
  if (ZSTD_isError(ZSTD_initDDict_internal(ddict, dict, dictSize,
                                           ZSTD_dlm_byRef, dictContentType)))
    return NULL;
  return ddict;
}

/*! ZSTD_estimateDDictSize() :
 *  Estimate amount of memory that will be needed to create a dictionary for
 * decompression. Note : dictionary created by reference using ZSTD_dlm_byRef
 * are smaller */
DEVICE_INLINE size_t
ZSTD_estimateDDictSize(size_t dictSize, ZSTD_dictLoadMethod_e dictLoadMethod) {
  return sizeof(ZSTD_DDict) + (dictLoadMethod == ZSTD_dlm_byRef ? 0 : dictSize);
}

DEVICE_INLINE size_t ZSTD_sizeof_DDict(const ZSTD_DDict *ddict) {
  if (ddict == NULL)
    return 0; /* support sizeof on NULL */
  return sizeof(*ddict) + (ddict->dictBuffer ? ddict->dictSize : 0);
}

/*! ZSTD_getDictID_fromDDict() :
 *  Provides the dictID of the dictionary loaded into `ddict`.
 *  If @return == 0, the dictionary is not conformant to Zstandard
 * specification, or empty. Non-conformant dictionaries can still be loaded, but
 * as content-only dictionaries. */
DEVICE_INLINE unsigned ZSTD_getDictID_fromDDict(const ZSTD_DDict *ddict) {
  if (ddict == NULL)
    return 0;
  return ddict->dictID;
}

} // namespace zstd
} // namespace hipcomp
