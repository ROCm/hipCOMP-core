/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// MIT License
//
// Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef HIPCOMP_H
#define HIPCOMP_H

#include <hip/hip_runtime.h>
#include "hipcomp/shared_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

#define HIPCOMP_MAJOR_VERSION 2
#define HIPCOMP_MINOR_VERSION 2
#define HIPCOMP_PATCH_VERSION 0

/* Supported datatypes */
typedef enum hipcompType_t
{
  HIPCOMP_TYPE_CHAR = 0,      // 1B
  HIPCOMP_TYPE_UCHAR = 1,     // 1B
  HIPCOMP_TYPE_SHORT = 2,     // 2B
  HIPCOMP_TYPE_USHORT = 3,    // 2B
  HIPCOMP_TYPE_INT = 4,       // 4B
  HIPCOMP_TYPE_UINT = 5,      // 4B
  HIPCOMP_TYPE_LONGLONG = 6,  // 8B
  HIPCOMP_TYPE_ULONGLONG = 7, // 8B
  HIPCOMP_TYPE_BITS = 0xff    // 1b
} hipcompType_t;

/******************************************************************************
 * FUNCTION PROTOTYPES ********************************************************
 *****************************************************************************/

/**
 * NOTE: These interfaces will be removed in future releases, please switch to
 * the compression schemes specific interfaces in hipcomp/cascaded.h,
 * hipcomp/lz4.h, hipcomp/snappy, hipcomp/bitcomp.h, and hipcomp/gdeflate.h.
 */

/**
 * DEPRECATED: Will be removed in future releases.
 *
 * @brief Extracts the metadata from the input in_ptr on the device and copies
 *it to the host.
 *
 * @param in_ptr The compressed memory on the device.
 * @param in_bytes The size of the compressed memory on the device.
 * @param metadata_ptr The metadata on the host to create from the compresesd
 * data.
 * @param stream The stream to use for reading memory from the device.
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompDecompressGetMetadata(
    const void* in_ptr,
    size_t in_bytes,
    void** metadata_ptr,
    hipStream_t stream);

/**
 * DEPRECATED: Will be removed in future releases.
 *
 * @brief Destroys the metadata object and frees the associated memory.
 *
 * @param metadata_ptr The pointer to destroy.
 */
void hipcompDecompressDestroyMetadata(void* metadata_ptr);

/**
 * DEPRECATED: Will be removed in future releases.
 *
 * @brief Computes the required temporary workspace required to perform
 * decompression.
 *
 * @para metadata_ptr The metadata.
 * @param temp_bytes The size of the required temporary workspace in bytes
 * (output).
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t
hipcompDecompressGetTempSize(const void* metadata_ptr, size_t* temp_bytes);

/**
 * DEPRECATED: Will be removed in future releases.
 *
 * @brief Computes the size of the uncompressed data in bytes.
 *
 * @para metadata_ptr The metadata.
 * @param output_bytes The size of the uncompressed data (output).
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t
hipcompDecompressGetOutputSize(const void* metadata_ptr, size_t* output_bytes);

/**
 * DEPRECATED: Will be removed in future releases.
 *
 * @brief Get the type of the compressed data.
 *
 * @param metadata_ptr The metadata.
 * @param type The data type (output).
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t
hipcompDecompressGetType(const void* metadata_ptr, hipcompType_t* type);

/**
 * DEPRECATED: Will be removed in future releases.
 *
 * @brief Perform the asynchronous decompression.
 *
 * @param in_ptr The compressed data on the device to decompress.
 * @param in_bytes The size of the compressed data.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace.
 * @param metadata_ptr The metadata.
 * @param out_ptr The output location on the device.
 * @param out_bytes The size of the output location.
 * @param stream The hip stream to operate on.
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompDecompressAsync(
    const void* in_ptr,
    size_t in_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    void* metadata_ptr,
    void* out_ptr,
    size_t out_bytes,
    hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif