/*
 * Copyright (c) 2017-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef HIPCOMP_SNAPPY_H
#define HIPCOMP_SNAPPY_H

#include "hipcomp.h"

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  int reserved;
} hipcompBatchedSnappyOpts_t;

static const hipcompBatchedSnappyOpts_t hipcompBatchedSnappyDefaultOpts = {0};

/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 *
 * @param num_chunks The number of items in the batch.
 * @param max_uncompressed_chunk_size The size of the largest chunk when uncompressed.
 * @param temp_bytes The amount of temporary GPU space that will be required to
 * decompress.
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompBatchedSnappyDecompressGetTempSize(
    size_t num_chunks, size_t max_uncompressed_chunk_size, size_t* temp_bytes);

/**
 * @brief Compute uncompressed sizes.
 *
 * @param device_compresed_ptrs The pointers on the GPU, to the compressed chunks.
 * @param device_compressed_bytes The size of each compressed chunk on the GPU.
 * @param device_uncompressed_bytes The actual size of each uncompressed chunk.
 * @param batch_size The number of chunks in the batch.
 * @param stream The HIP stream to operate on.
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompBatchedSnappyGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream);

/**
 * @brief Perform decompression.
 *
 * @param device_compresed_ptrs The pointers on the GPU, to the compressed chunks.
 * @param device_compressed_bytes The size of each compressed chunk on the GPU.
 * @param device_uncompressed_bytes The size of each device_uncompressed_ptr[i] buffer.
 * @param device_actual_uncompressed_bytes The actual size of each uncompressed chunk
 * Can be nullptr if desired, in which case the actual_uncompressed_bytes is not reported.
 * @param batch_size The number of chunks in the batch.
 * @param device_temp_ptr The temporary GPU space, could be NULL in case temprorary space is not needed.
 * @param temp_bytes The size of the temporary GPU space.
 * @param device_uncompressed_ptr The pointers on the GPU, to where to uncompress each chunk (output).
 * @param device_statuses The pointers on the GPU, to where to uncompress each chunk (output).
 * Can be nullptr if desired, in which case error status is not reported.
 * @param stream The HIP stream to operate on.
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompBatchedSnappyDecompressAsync(
    const void* const* device_compresed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    const size_t temp_bytes,
    void* const* device_uncompressed_ptr,
    hipcompStatus_t* device_statuses,
    hipStream_t stream);

/**
 * @brief Get temporary space required for compression.
 *
 * @param batch_size The number of items in the batch.
 * @param max_chunk_size The maximum size of a chunk in the batch.
 * @param format_ops Snappy compression options.
 * @param temp_bytes The size of the required GPU workspace for compression
 * (output).
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompBatchedSnappyCompressGetTempSize(
    size_t batch_size,
    size_t max_chunk_size,
    hipcompBatchedSnappyOpts_t format_ops,
    size_t* temp_bytes);

/**
 * @brief Get the maximum size any chunk could compress to in the batch. That
 * is, the minimum amount of output memory required to be given
 * hipcompBatchedSnappyCompressAsync() for each batch item.
 *
 * @param max_chunk_size The maximum size of a chunk in the batch.
 * @param format_ops Snappy compression options.
 * @param max_compressed_size The maximum compressed size of the largest chunk
 * (output).
 *
 * @return The hipcompSuccess unless there is an error.
 */
hipcompStatus_t hipcompBatchedSnappyCompressGetMaxOutputChunkSize(
    size_t max_chunk_size,
    hipcompBatchedSnappyOpts_t format_opts,
    size_t* max_compressed_size);

/**
 * @brief Perform compression.
 *
 * The caller is responsible for passing device_compressed_bytes of size
 * sufficient to hold compressed data
 *
 * @param device_uncompressed_ptr The pointers on the GPU, to uncompressed batched items.
 * @param device_uncompressed_bytes The size of each uncompressed batch item on the GPU.
 * @param max_uncompressed_chunk_bytes The size of the largest uncompressed chunk.
 * @param batch_size The number of chunks in the batch.
 * @param device_temp_ptr The temporary GPU workspace, could be NULL in case temprorary space is not needed.
 * @param temp_bytes The size of the temporary GPU workspace.
 * @param device_compressed_ptr The pointers on the GPU, to the output location for each compressed batch item (output).
 * @param device_compressed_bytes The compressed size of each chunk on the GPU (output).
 * @param format_ops Snappy compression options.
 * @param stream The HIP stream to operate on.
 *
 * @return hipcompSuccess if successfully launched, and an error code otherwise.
 */
hipcompStatus_t hipcompBatchedSnappyCompressAsync(
    const void* const* device_uncompressed_ptr,
    const size_t* device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    void* device_temp_ptr,
    size_t temp_bytes,
    void* const* device_compressed_ptr,
    size_t* device_compressed_bytes,
    hipcompBatchedSnappyOpts_t format_ops,
    hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif