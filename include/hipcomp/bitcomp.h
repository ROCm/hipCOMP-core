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

#ifndef HIPCOMP_BITCOMP_H
#define HIPCOMP_BITCOMP_H

#include "hipcomp.h"

#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Structure for configuring Bitcomp compression.
 */
typedef struct
{
  /**
   * @brief Bitcomp algorithm options.
   *  algorithm_type: The type of Bitcomp algorithm used.
   *    0 : Default algorithm, usually gives the best compression ratios
   *    1 : "Sparse" algorithm, works well on sparse data (with lots of zeroes).
   *        and is usually a faster than the default algorithm.
   */
  int algorithm_type;
} hipcompBitcompFormatOpts;

static const hipcompBitcompFormatOpts hipcompBitcompDefaultOpts = {0};

/**
 * @brief Get the temporary workspace size required to perform compression.
 *
 * @param format_opts The bitcomp format options (can pass NULL for default
 * options).
 * @param in_type The type of the uncompressed data.
 * @param uncompressed_bytes The size of the uncompressed data in bytes.
 * @param temp_bytes The size of the required temporary workspace in bytes
 * (output).
 * @param max_compressed_bytes The maximum size of the compressed data
 * (output).
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompBitcompCompressConfigure(
    const hipcompBitcompFormatOpts* opts,
    hipcompType_t in_type,
    size_t in_bytes,
    size_t* metadata_bytes,
    size_t* temp_bytes,
    size_t* max_compressed_bytes);

/**
 * @brief Perform asynchronous compression.
 *
 * @param format_opts The bitcomp format options (can pass NULL for default
 * options).
 * @param in_type The data type of the uncompressed data.
 * @param uncompressed_ptr The uncompressed data on the device.
 * @param uncompressed_bytes The size of the uncompressed data in bytes.
 * @param temp_ptr The temporary workspace on the device.
 * @param temp_bytes The size of the temporary workspace in bytes.
 * @param compressed_ptr The location to write compresesd data to on the device
 * (output).
 * @param compressed_bytes The size of the compressed data (output). This must
 * be GPU accessible.
 * @param stream The hip stream to operate on.
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompBitcompCompressAsync(
    const hipcompBitcompFormatOpts* format_opts,
    hipcompType_t in_type,
    const void* uncompressed_ptr,
    size_t uncompressed_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    void* compressed_ptr,
    size_t* compressed_bytes,
    hipStream_t stream);

/**
 * @brief Extracts the metadata from the input in_ptr on the device and copies
 * it to the host. This function synchronizes on the stream.
 *
 * @param compressed_ptr The compressed memory on the device.
 * @param compressed_bytes The size of the compressed memory on the device.
 * @param metadata_ptr The metadata on the host to create from the compresesd
 * data (output).
 * @param metadata_bytes The size of the created metadata (output).
 * @param temp_bytes The amount of temporary space required for decompression
 * (output).
 * @param uncompressed_bytes The size the data will decompress to (output).
 * @param stream The stream to use for copying from the device to the host.
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompBitcompDecompressConfigure(
    const void* compressed_ptr,
    size_t compressed_bytes,
    void** metadata_ptr,
    size_t* metadata_bytes,
    size_t* temp_bytes,
    size_t* uncompressed_bytes,
    hipStream_t stream);

/**
 * @brief Destroys the metadata object and frees the associated memory.
 *
 * @param metadata_ptr The pointer to destroy.
 */
void hipcompBitcompDestroyMetadata(void* metadata_ptr);

/**
 * @brief Perform the asynchronous decompression.
 *
 * @param compressed_ptr The compressed data on the device to decompress.
 * @param compressed_bytes The size of the compressed data.
 * @param metadata_ptr The metadata.
 * @param metadata_bytes The size of the metadata.
 * @param temp_ptr The temporary workspace on the device. Not used, can pass
 * NULL.
 * @param temp_bytes The size of the temporary workspace. Not used.
 * @param uncompressed_ptr The output location on the device.
 * @param uncompressed_bytes The size of the output location.
 * @param stream The hip stream to operate on.
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompBitcompDecompressAsync(
    const void* compressed_ptr,
    size_t compressed_bytes,
    void* metadata_ptr,
    size_t metadata_bytes,
    void* temp_ptr,
    size_t temp_bytes,
    void* uncompressed_ptr,
    size_t uncompressed_bytes,
    hipStream_t stream);

/**
 * @brief Checks if the compressed data was compressed with bitcomp.
 *
 * @param in_ptr The compressed data.
 * @param in_bytes The size of the compressed buffer.
 *
 * @return 1 if the data was compressed with bitcomp, 0 otherwise
 */
int hipcompIsBitcompData(const void* const in_ptr, size_t in_bytes);

/******************************************************************************
 * Batched compression/decompression interface
 *****************************************************************************/

/**
 * @brief Structure for configuring Bitcomp compression.
 */
typedef struct
{
  /**
   * @brief Bitcomp algorithm options.
   *  algorithm_type: The type of Bitcomp algorithm used.
   *    0 : Default algorithm, usually gives the best compression ratios
   *    1 : "Sparse" algorithm, works well on sparse data (with lots of zeroes).
   *        and is usually a faster than the default algorithm.
   *  data_type is one of hipcomp's possible data types
   */
  int algorithm_type;
  hipcompType_t data_type;
} hipcompBatchedBitcompFormatOpts;

static const hipcompBatchedBitcompFormatOpts hipcompBatchedBitcompDefaultOpts
    = {0, HIPCOMP_TYPE_UCHAR};

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
hipcompStatus_t hipcompBatchedBitcompCompressGetMaxOutputChunkSize(
    size_t max_chunk_size,
    hipcompBatchedBitcompFormatOpts format_opts,
    size_t* max_compressed_size);
/**
 * @brief Perform batched asynchronous compression.
 *
 * NOTE: The maximum number of batch partitions is 2^31.
 * 
 * NOTE: Unlike `hipcompBitcompCompressAsync`, a valid compression format must
 * be supplied to `format_opts`.
 *
 * @param[in] device_uncompressed_ptrs Array with size \p batch_size of pointers
 * to the uncompressed partitions. Both the pointers and the uncompressed data
 * should reside in device-accessible memory. The uncompressed data must start
 * at locations with alignments of the data type.
 * @param[in] device_uncompressed_bytes Sizes of the uncompressed partitions in
 * bytes. The sizes should reside in device-accessible memory.
 * @param[in] max_uncompressed_chunk_bytes This argument is not used.
 * @param[in] batch_size Number of partitions to compress.
 * @param[in] device_temp_ptr This argument is not used.
 * @param[in] temp_bytes This argument is not used.
 * @param[out] device_compressed_ptrs Array with size \p batch_size of pointers
 * to the output compressed buffers. Both the pointers and the compressed
 * buffers should reside in device-accessible memory. Each compressed buffer
 * should be preallocated with size at least (8B + the uncompressed size). Each
 * compressed buffer should start at a location with 64-bit alignment.
 * @param[out] device_compressed_bytes Compressed sizes in bytes for all
 * partitions. The buffer should be preallocated in device-accessible memory.
 * @param[in] format_opts The bitcomp format options. The format must be valid.
 * @param[in] type The data type of the uncompressed data.
 * @param[in] stream The hip stream to operate on.
 *
 * @return hipcompSuccess if successful, and an error code otherwise.
 */
hipcompStatus_t hipcompBatchedBitcompCompressAsync(
    const void* const* device_uncompressed_ptrs,
    const size_t* device_uncompressed_bytes,
    size_t max_uncompressed_chunk_bytes, // not used
    size_t batch_size,
    void* device_temp_ptr, // not used
    size_t temp_bytes,     // not used
    void* const* device_compressed_ptrs,
    size_t* device_compressed_bytes,
    const hipcompBatchedBitcompFormatOpts format_opts,
    hipStream_t stream);

/**
 * @brief Perform batched asynchronous decompression.
 *
 * NOTE: This function is used to decompress compressed buffers produced by
 * `hipcompBatchedBitcompCompressAsync`. It can also decompress buffers
 * compressed with `hipcompBitcompCompressAsync` or the standalone Bitcomp library.
 * 
 * NOTE: The function is not completely asynchronous, as it needs to look
 * at the compressed data in order to create the proper bitcomp handle.
 * The stream is synchronized, the data is examined, then the asynchronous
 * decompression is launched.
 *
 * @param[in] device_compressed_ptrs Array with size \p batch_size of pointers
 * in device-accessible memory to compressed buffers. Each compressed buffer
 * should reside in device-accessible memory and start at a location which is
 * 64-bit aligned.
 * @param[in] device_compressed_bytes This argument is not used.
 * @param[in] device_uncompressed_bytes Sizes of the output uncompressed
 * buffers in bytes. The sizes should reside in device-accessible memory. If the
 * size is not large enough to hold all decompressed elements, the decompressor
 * will set the status specified in \p device_statuses corresponding to the
 * overflow partition to `hipcompErrorCannotDecompress`.
 * @param[out] device_actual_uncompressed_bytes Array with size \p batch_size of
 * the actual number of bytes decompressed for every partitions. This argument
 * needs to be preallocated.
 * @param[in] batch_size Number of partitions to decompress.
 * @param[in] device_temp_ptr This argument is not used.
 * @param[in] temp_bytes This argument is not used.
 * @param[out] device_uncompressed_ptrs This argument is not used.
 * @param[out] device_statuses Array with size \p batch_size of statuses in
 * device-accessible memory. This argument needs to be preallocated. For each
 * partition, if the decompression is successful, the status will be set to
 * `hipcompSuccess`. If the decompression is not successful, for example due to
 * the corrupted input or out-of-bound errors, the status will be set to
 * `hipcompErrorCannotDecompress`.
 * @param[in] stream The hip stream to operate on.
 */
hipcompStatus_t hipcompBatchedBitcompDecompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes, // not used
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr, // not used
    size_t temp_bytes,           // not used
    void* const* device_uncompressed_ptrs,
    hipcompStatus_t* device_statuses,
    hipStream_t stream);

/**
 * @brief Asynchronously get the number of bytes of the uncompressed data in
 * every partitions.
 *
 * @param[in] device_compressed_ptrs Array with size \p batch_size of pointers
 * in device-accessible memory to compressed buffers.
 * @param[in] device_compressed_bytes This argument is not used.
 * @param[out] device_uncompressed_bytes Sizes of the uncompressed data in
 * bytes. If there is an error when retrieving the size of a partition, the
 * uncompressed size of that partition will be set to 0. This argument needs to
 * be prealloated in device-accessible memory.
 * @param[in] batch_size Number of partitions to check sizes.
 * @param[in] stream The hip stream to operate on.
 */
hipcompStatus_t hipcompBatchedBitcompGetDecompressSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream);

/**
 * @brief Return the temp size needed for Bitcomp compression.
 * Bitcomp currently doesn't use any temp memory.
 * 
 * @param[in] batch_size  Number of chunks
 * @param[in] max_chunk_bytes Size in bytes of the largest chunk
 * @param[in] format_opts Bitcomp options
 * @param[out] temp_bytes The temp size
 */
hipcompStatus_t hipcompBatchedBitcompCompressGetTempSize(
    size_t batch_size,
    size_t max_chunk_bytes,
    hipcompBatchedBitcompFormatOpts format_opts,
    size_t * temp_bytes);

/**
 * @brief Return the temp size needed for Bitcomp decompression.
 * Bitcomp currently doesn't use any temp memory.
 * 
 * @param[in] batch_size  Number of chunks
 * @param[in] max_chunk_bytes Size in bytes of the largest chunk
 * @param[in] format_opts Bitcomp options
 * @param[out] temp_bytes The temp size
 */
hipcompStatus_t hipcompBatchedBitcompDecompressGetTempSize(
    size_t batch_size,
    size_t max_chunk_bytes,
    size_t * temp_bytes);

#ifdef __cplusplus
}
#endif

#endif