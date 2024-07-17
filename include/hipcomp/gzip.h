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
// Modifications Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights
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

#ifndef HIPCOMP_GZIP_H
#define HIPCOMP_GZIP_H

#include "hipcomp.h"

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/******************************************************************************
 * Batched decompression interface for gzip
 *****************************************************************************/

/**
 * @brief Get the amount of temp space required on the GPU for decompression.
 *
 * @param num_chunks The number of items in the batch.
 * @param max_uncompressed_chunk_bytes The size of the largest chunk in bytes
 * when uncompressed.
 * @param temp_bytes The amount of temporary GPU space that will be required to
 * decompress.
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompBatchedGzipDecompressGetTempSize(
    size_t num_chunks, size_t max_uncompressed_chunk_bytes, size_t *temp_bytes);

/**
 * @brief Perform decompression asynchronously.
 *
 * All pointers must be GPU accessible.
 *
 * In case that a chunk of compressed data is not a valid GZIP stream,
 * hipcompErrorCannotDecompress will be flagged for that chunk.
 * In case that a compressed chunk cannot be decompressed because the output
 * buffer is too small, hipcompErrorOutOfMemory will be flagged for that chunk.
 *
 * @note If ``hipcompErrorOutOfMemory`` is detected in one of the blocks
 *
 * Example usage:
 *
 * ```c++
 * #include "hipcomp/gzip.h"
 * #include "hipcomp/helpers.h"
 *
 * int main() {
 *   hipStream_t stream = nullptr;
 *   hipStreamCreate(&stream)
 *
 *   const int batch_size = ...;
 *
 *   // device arrays
 *   void** device_compressed_ptrs = nullptr;
 *   size_t* device_compressed_bytes = nullptr;
 *   void** device_uncompressed_ptrs = nullptr;
 *   size_t* device_uncompressed_bytes = nullptr;
 *   size_t* device_actual_uncompressed_bytes = nullptr;
 *   hipcompStatus_t* device_statuses = nullptr;
 *
 *   // allocate all device arrays (~batch_size) and initialize them
 *   // ...
 *
 *   hipcompBatchedGzipDecompressAsync( // from "hipcomp/gzip.h"
 *     device_compressed_ptrs,
 *     device_compressed_bytes,
 *     device_uncompressed_bytes,
 *     device_actual_uncompressed_bytes,
 *     batch_size,
 *     nullptr, // device_temp_ptr
 *     0, // temp_bytes
 *     device_uncompressed_ptrs,
 *     device_statuses,
 *     stream);
 *
 *   // optional: do some other work before synchronizing the stream
 *
 *   // only after synchronizing the stream, the blocks are done with their
 * computations hipStreamSynchronize(stream);
 *
 *   // now that we have `device_actual_uncompressed_bytes`, reallocate the
 * undersized decompression buffers int oom_count = 0;
 *   CHECK_API_CALL(hipcompReallocateDecompressionBuffers(&oom_count, // will be
 * modified device_uncompressed_ptrs,  // will be modified
 *                                                        device_uncompressed_bytes,
 * // will be modified device_actual_uncompressed_bytes, batch_size));
 *
 *   // rerun if any out-of-memory error has been reported (and the
 * corresponding buffers reallocated) if ( oom_count > 0 ) {
 *     hipcompBatchedGzipDecompressAsync(
 *       device_compressed_ptrs,
 *       device_compressed_bytes,
 *       device_uncompressed_bytes,
 *       device_actual_uncompressed_bytes,
 *       batch_size,
 *       nullptr, // not needed, device_temp_ptr
 *       0, // temp_bytes
 *       device_uncompressed_ptrs,
 *       device_statuses,
 *       stream);
 *   }
 * }
 * ```
 *
 * @param device_compressed_ptrs The pointers on the GPU, to the compressed
 * chunks.
 * @param device_compressed_bytes The size of each compressed chunk on the GPU.
 * This pointer must be GPU accessible.
 * @param device_uncompressed_bytes The decompressed buffer size. This is needed
 * to prevent OOB accesses. This pointer must be GPU accessible.
 * @param device_actual_uncompressed_bytes The actual calculated decompressed
 * size of each chunk. Can be nullptr if desired,
 * in which case the actual_uncompressed_bytes is not reported.
 * @param batch_size The number of batch items.
 * @param device_temp_ptr The temporary GPU space.
 * @param temp_bytes The size of the temporary GPU space.
 * @param device_uncompressed_ptrs The pointers on the GPU, to where to
 * uncompress each chunk (output).
 * @param device_statuses The status for each chunk of whether it was
 * decompressed or not. Can be nullptr if desired,
 * in which case error status is not reported.
 * @param stream The HIP stream to operate on.
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompBatchedGzipDecompressAsync(
    const void *const *device_compressed_ptrs,
    const size_t *device_compressed_bytes,
    const size_t *device_uncompressed_bytes,
    size_t *device_actual_uncompressed_bytes, size_t batch_size,
    void *const device_temp_ptr, size_t temp_bytes,
    void *const *device_uncompressed_ptrs, hipcompStatus_t *device_statuses,
    hipStream_t stream);

/**
 * @brief Calculates the decompressed size of each chunk asynchronously. This is
 * needed when we do not know the expected output size. All pointers must be GPU
 * accessible. Note, if the stream is corrupt, the sizes will be garbage.
 *
 * @param device_compress_ptrs The compressed chunks of data. List of pointers
 * must be GPU accessible along with each chunk.
 * @param device_compressed_bytes The size of each compressed chunk. Must be GPU
 * accessible.
 * @param device_uncompressed_bytes The calculated decompressed size of each
 * chunk. Must be GPU accessible.
 * @param batch_size The number of chunks
 * @param stream The HIP stream to operate on.
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompBatchedGzipGetDecompressSizeAsync(
    const void *const *device_compressed_ptrs,
    const size_t *device_compressed_bytes, size_t *device_uncompressed_bytes,
    size_t batch_size, hipStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // HIPCOMP_GZIP_H
