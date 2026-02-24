/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include "common.h"
#include "hip/hip_runtime_api.h"
#include "hipcomp.h"

namespace hipcomp {

// TODO(HIP/AMD): ZStd compression not implemented in hipCOMP
// /**
//  * @brief Interface for compressing data with ZSTD
//  *
//  * The function compresses multiple independent chunks of data.
//  * All the pointers parameters are to GPU-accessible memory.
//  *
//  * @param[in] device_in_ptr Pointer to the list of pointers to
//  * the GPU-accessible uncompressed data.
//  * @param[in] device_in_bytes Pointer to the list of sizes of uncompressed
//  * data
//  * @param[in] device_out_ptr Pointer to the buffer with pointers,
//  * where the function should put compressed data to.
//  * @param[in] device_out_available_bytes Pointer to the list of sizes of
//  * memory chunks referenced by device_out_ptr. Could be null-ptr indicating
//  * all output buffers has enough size to store compressed data.
//  * @param[out] outputs Pointer to the statuses of compression for each chunk.
//  * Could be null-ptr.
//  * @param[out] device_out_bytes Pointer to the list of actual sizes
//  * of compressed data.
//  * @param[in] count The number of chunks to compress.
//  * @param[in] stream All the compression will be enqueued into this HIP
//  * stream and run asynchronously.
//  **/
// void gpu_zstd_compress(
//   const void* const* device_in_ptr,
// 	const size_t* device_in_bytes,
// 	void* const* device_out_ptr,
// 	const size_t* device_out_available_bytes,
// 	gpu_gzip_status_s *outputs,
// 	size_t* device_out_bytes,
//   int count,
//   hipStream_t stream);

/**
 * @brief Interface for decompressing data with ZSTD
 *
 * The function decompresses multiple independent chunks of data.
 * All the pointers parameters are to GPU-accessible memory.
 *
 * @param[in] device_in_ptr Pointer to the list of pointers to
 * the GPU-accessible compressed data.
 * @param[in] device_in_bytes Pointer to the list of sizes of compressed
 * data.
 * @param[in] device_out_ptr Pointer to the buffer with pointers,
 * where the function should put uncompressed data to.
 * @param[in] device_out_available_bytes Pointer to the list of sizes of
 * memory chunks referenced by device_out_ptr. Could be null-ptr indicating
 * all output buffers has enough size to store uncompressed data.
 * @param[out] status Pointer to the statuses of decompression for each chunk.
 * Could be null-ptr.
 * @param[out] device_out_bytes Pointer to the list of actual sizes
 * of uncompressed data. Could be null-ptr.
 * @param[in] count The number of chunks to decompress.
 * @param[in] stream All the decompression will be enqueued into this HIP
 * stream and run asynchronously.
 **/
void gpu_zstd_decompress(const void *const *device_in_ptr,
                         const size_t *device_in_bytes,
                         void *const *device_out_ptr,
                         const size_t *device_out_available_bytes,
                         hipcompStatus_t *status, size_t *device_out_bytes,
                         int count, hipStream_t stream);

/**
 * @brief Get uncompressed sizes for ZSTD compressed chunks.
 *
 * This function reads the ZSTD frame headers to determine the actual
 * uncompressed size for each chunk. This is more accurate than estimation
 * as it uses ZSTD's internal functions to parse the frame metadata.
 *
 * @param[in] device_in_ptr Pointer to the list of pointers to
 * the GPU-accessible compressed data.
 * @param[in] device_in_bytes Pointer to the list of sizes of compressed
 * data.
 * @param[out] device_out_bytes Pointer to the list where uncompressed sizes
 * will be stored. The function will put 0 for the uncompressed size if it
 * detects an error reading the ZSTD frame header.
 * @param[in] count The number of chunks to process.
 * @param[in] stream All the computations will be enqueued into this HIP
 * stream and run asynchronously.
 **/
void gpu_zstd_get_uncompressed_sizes(const void *const *device_in_ptr,
                                     const size_t *device_in_bytes,
                                     size_t *device_out_bytes, int count,
                                     hipStream_t stream);

} // namespace hipcomp
