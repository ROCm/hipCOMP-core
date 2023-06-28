/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#pragma once

#include "hip/hip_runtime.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// NOTE: this is for testing the C API, and thus must only contain features of
// C, not C++.

#define REQUIRE(a)                                                             \
  do {                                                                         \
    if (!(a)) {                                                                \
      printf("Check " #a " at %d failed.\n", __LINE__);                        \
      return FAIL_TEST;                                                        \
    }                                                                          \
  } while (0)

#define HIP_CHECK(func)                                                       \
  do {                                                                         \
    hipError_t rt = (func);                                                   \
    if (rt != hipSuccess) {                                                   \
      printf(                                                                  \
          "API call failure \"" #func "\" with %d at " __FILE__ ":%d\n",       \
          (int)rt,                                                             \
          __LINE__);                                                           \
      return FAIL_TEST;                                                        \
    }                                                                          \
  } while (0)

// There's a lot of redundancy in these macros, but if there's a mismatch, it
// will show up at compile time, and things should only need to change if our
// interface changes, which should be very infrequent.
#define GENERATE_TESTS(NAME)                                                   \
  hipcompStatus_t compressGetTempSize(                                          \
      const size_t batch_size,                                                 \
      const size_t max_uncompressed_chunk_bytes,                               \
      size_t* const temp_bytes)                                                \
  {                                                                            \
    return hipcompBatched##NAME##CompressGetTempSize(                           \
        batch_size,                                                            \
        max_uncompressed_chunk_bytes,                                          \
        hipcompBatched##NAME##DefaultOpts,                                      \
        temp_bytes);                                                           \
  }                                                                            \
  hipcompStatus_t compressGetMaxOutputChunkSize(                                \
      const size_t max_uncompressed_chunk_bytes,                               \
      size_t* const max_compressed_bytes)                                      \
  {                                                                            \
    return hipcompBatched##NAME##CompressGetMaxOutputChunkSize(                 \
        max_uncompressed_chunk_bytes,                                          \
        hipcompBatched##NAME##DefaultOpts,                                      \
        max_compressed_bytes);                                                 \
  }                                                                            \
  hipcompStatus_t compressAsync(                                                \
      const void* const* const device_in_ptr,                                  \
      const size_t* const device_in_bytes,                                     \
      const size_t max_uncompressed_chunk_bytes,                               \
      const size_t batch_size,                                                 \
      void* const device_device_temp_ptr,                                      \
      const size_t temp_bytes,                                                 \
      void* const* device_out_ptr,                                             \
      size_t* const device_out_bytes,                                          \
      hipStream_t stream)                                                     \
  {                                                                            \
    return hipcompBatched##NAME##CompressAsync(                                 \
        device_in_ptr,                                                         \
        device_in_bytes,                                                       \
        max_uncompressed_chunk_bytes,                                          \
        batch_size,                                                            \
        device_device_temp_ptr,                                                \
        temp_bytes,                                                            \
        device_out_ptr,                                                        \
        device_out_bytes,                                                      \
        hipcompBatched##NAME##DefaultOpts,                                      \
        stream);                                                               \
  }                                                                            \
  hipcompStatus_t decompressGetSizeAsync(                                       \
      const void* const* const device_compressed_ptrs,                         \
      const size_t* const device_compressed_bytes,                             \
      size_t* const device_uncompressed_bytes,                                 \
      const size_t batch_size,                                                 \
      hipStream_t stream)                                                     \
  {                                                                            \
    return hipcompBatched##NAME##GetDecompressSizeAsync(                        \
        device_compressed_ptrs,                                                \
        device_compressed_bytes,                                               \
        device_uncompressed_bytes,                                             \
        batch_size,                                                            \
        stream);                                                               \
  }                                                                            \
  hipcompStatus_t decompressGetTempSize(                                        \
      const size_t num_chunks,                                                 \
      const size_t max_uncompressed_chunk_bytes,                               \
      size_t* const temp_bytes)                                                \
  {                                                                            \
    return hipcompBatched##NAME##DecompressGetTempSize(                         \
        num_chunks, max_uncompressed_chunk_bytes, temp_bytes);                 \
  }                                                                            \
  hipcompStatus_t decompressAsync(                                              \
      const void* const* device_compressed_ptrs,                               \
      const size_t* device_compressed_bytes,                                   \
      const size_t* device_uncompressed_bytes,                                 \
      size_t* device_actual_uncompressed_bytes,                                \
      size_t batch_size,                                                       \
      void* const device_temp_ptr,                                             \
      size_t temp_bytes,                                                       \
      void* const* device_uncompressed_ptrs,                                   \
      hipcompStatus_t* device_status_ptr,                                       \
      hipStream_t stream)                                                     \
  {                                                                            \
    return hipcompBatched##NAME##DecompressAsync(                               \
        device_compressed_ptrs,                                                \
        device_compressed_bytes,                                               \
        device_uncompressed_bytes,                                             \
        device_actual_uncompressed_bytes,                                      \
        batch_size,                                                            \
        device_temp_ptr,                                                       \
        temp_bytes,                                                            \
        device_uncompressed_ptrs,                                              \
        device_status_ptr,                                                     \
        stream);                                                               \
  }                                                                            \
  typedef int __hipcomp_semicolon_catch

// Declear the test function wrappers
hipcompStatus_t compressGetTempSize(
    const size_t batch_size,
    const size_t max_uncompressed_chunk_bytes,
    size_t* const temp_bytes);

hipcompStatus_t compressGetMaxOutputChunkSize(
    const size_t max_uncompressed_chunk_bytes,
    size_t* const max_compressed_bytes);

hipcompStatus_t compressAsync(
    const void* const* device_in_ptr,
    const size_t* device_in_bytes,
    size_t max_uncompressed_chunk_bytes,
    size_t batch_size,
    void* device_device_temp_ptr,
    size_t temp_bytes,
    void* const* device_out_ptr,
    size_t* device_out_bytes,
    hipStream_t stream);

hipcompStatus_t decompressGetSizeAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    size_t* device_uncompressed_bytes,
    size_t batch_size,
    hipStream_t stream);

hipcompStatus_t decompressGetTempSize(
    const size_t num_chunks,
    const size_t max_uncompressed_chunk_bytes,
    size_t* const temp_bytes);

hipcompStatus_t decompressAsync(
    const void* const* device_compressed_ptrs,
    const size_t* device_compressed_bytes,
    const size_t* device_uncompressed_bytes,
    size_t* device_actual_uncompressed_bytes,
    size_t batch_size,
    void* const device_temp_ptr,
    size_t temp_bytes,
    void* const* device_uncompressed_ptrs,
    hipcompStatus_t* device_status_ptrs,
    hipStream_t stream);

static const int PASS_TEST = 1;
static const int FAIL_TEST = 0;

int test_generic_batch_compression_and_decompression(
    const size_t batch_size, const size_t min_size, 
    const size_t max_size, const int support_nullptr)
{
  typedef int T;

  // set a constant seed
  srand(0);

  // prepare input and output on host
  size_t* host_batch_sizes = malloc(batch_size * sizeof(size_t));
  for (size_t i = 0; i < batch_size; ++i) {
    if (max_size > min_size) {
      host_batch_sizes[i] = (rand() % (max_size - min_size)) + min_size;
    } else if (max_size == min_size) {
      host_batch_sizes[i] = max_size;
    } else {
      printf("Invalid max_size (%zu) / min_size (%zu)\n", max_size, min_size);
      return FAIL_TEST;
    }
  }

  size_t* host_batch_bytes = malloc(batch_size * sizeof(size_t));
  size_t max_chunk_size = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    host_batch_bytes[i] = sizeof(T) * host_batch_sizes[i];
    if (host_batch_bytes[i] > max_chunk_size) {
      max_chunk_size = host_batch_bytes[i];
    }
  }

  T** host_input = malloc(sizeof(T*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    host_input[i] = malloc(sizeof(T) * host_batch_sizes[i]);
    for (size_t j = 0; j < host_batch_sizes[i]; ++j) {
      // make sure there should be some repeats to compress
      host_input[i][j] = (rand() % 4) + 300;
    }
  }
  free(host_batch_sizes);

  T** host_output = malloc(sizeof(T*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    host_output[i] = malloc(host_batch_bytes[i]);
  }

  // prepare gpu buffers
  void** host_in_ptrs = malloc(sizeof(void*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    HIP_CHECK(hipMalloc(&host_in_ptrs[i], host_batch_bytes[i]));
    HIP_CHECK(hipMemcpy(
        host_in_ptrs[i],
        host_input[i],
        host_batch_bytes[i],
        hipMemcpyHostToDevice));
  }
  void** device_in_pointers;
  HIP_CHECK(hipMalloc(
      (void**)&device_in_pointers, sizeof(*device_in_pointers) * batch_size));
  HIP_CHECK(hipMemcpy(
      device_in_pointers,
      host_in_ptrs,
      sizeof(*device_in_pointers) * batch_size,
      hipMemcpyHostToDevice));

  size_t* device_batch_bytes;
  HIP_CHECK(hipMalloc(
      (void**)&device_batch_bytes, sizeof(*device_batch_bytes) * batch_size));
  HIP_CHECK(hipMemcpy(
      device_batch_bytes,
      host_batch_bytes,
      sizeof(*device_batch_bytes) * batch_size,
      hipMemcpyHostToDevice));

  hipcompStatus_t status;

  // Compress on the GPU using batched API
  size_t comp_temp_bytes;
  status = compressGetTempSize(batch_size, max_chunk_size, &comp_temp_bytes);
  if (max_chunk_size > 1<<16) printf("max_chunk_size = %zu\n", max_chunk_size);
  REQUIRE(status == hipcompSuccess);

  void* d_comp_temp;
  HIP_CHECK(hipMalloc(&d_comp_temp, comp_temp_bytes));

  size_t max_comp_out_bytes;
  status = compressGetMaxOutputChunkSize(max_chunk_size, &max_comp_out_bytes);
  REQUIRE(status == hipcompSuccess);

  void** host_comp_out = malloc(sizeof(void*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    HIP_CHECK(hipMalloc(&host_comp_out[i], max_comp_out_bytes));
  }
  void** device_comp_out;
  HIP_CHECK(hipMalloc(
      (void**)&device_comp_out, sizeof(*device_comp_out) * batch_size));
  HIP_CHECK(hipMemcpy(
      device_comp_out,
      host_comp_out,
      sizeof(*device_comp_out) * batch_size,
      hipMemcpyHostToDevice));

  size_t* device_comp_out_bytes;
  HIP_CHECK(hipMalloc(
      (void**)&device_comp_out_bytes,
      sizeof(*device_comp_out_bytes) * batch_size));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  status = compressAsync(
      (const void* const*)device_in_pointers,
      device_batch_bytes,
      max_chunk_size,
      batch_size,
      d_comp_temp,
      comp_temp_bytes,
      device_comp_out,
      device_comp_out_bytes,
      stream);
  REQUIRE(status == hipcompSuccess);
  HIP_CHECK(hipStreamSynchronize(stream));

  HIP_CHECK(hipFree(d_comp_temp));
  for (size_t i = 0; i < batch_size; ++i) {
    HIP_CHECK(hipFree(host_in_ptrs[i]));
  }
  hipFree(device_in_pointers);
  free(host_in_ptrs);

  size_t temp_bytes;
  status = decompressGetTempSize(batch_size, max_chunk_size, &temp_bytes);

  void* device_temp_ptr;
  HIP_CHECK(hipMalloc(&device_temp_ptr, temp_bytes));

  size_t* device_decomp_out_bytes;
  HIP_CHECK(hipMalloc(
      (void**)&device_decomp_out_bytes,
      sizeof(*device_decomp_out_bytes) * batch_size));

  status = decompressGetSizeAsync(
      (const void* const*)device_comp_out,
      device_comp_out_bytes,
      device_decomp_out_bytes,
      batch_size,
      stream);
  REQUIRE(status == hipcompSuccess);
  HIP_CHECK(hipStreamSynchronize(stream));

  // copy the output sizes down and check them
  size_t* host_decomp_bytes = malloc(sizeof(size_t) * batch_size);
  HIP_CHECK(hipMemcpy(
      host_decomp_bytes,
      device_decomp_out_bytes,
      sizeof(*host_decomp_bytes) * batch_size,
      hipMemcpyDeviceToHost));
  for (size_t i = 0; i < batch_size; ++i) {
    REQUIRE(host_decomp_bytes[i] == host_batch_bytes[i]);
  }

  void** host_decomp_out = malloc(sizeof(void*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    HIP_CHECK(hipMalloc(&host_decomp_out[i], host_batch_bytes[i]));
  }
  void** device_decomp_out;
  hipMalloc(
      (void**)&device_decomp_out, sizeof(*device_decomp_out) * batch_size);
  HIP_CHECK(hipMemcpy(
      device_decomp_out,
      host_decomp_out,
      sizeof(*device_decomp_out) * batch_size,
      hipMemcpyHostToDevice));

  // Test functionality with null device_statuses and device_decomp_out_bytes
  if (support_nullptr)
  {
    status = decompressAsync(
        (const void* const*)device_comp_out,
        device_comp_out_bytes,
        device_batch_bytes,
        NULL,
        batch_size,
        device_temp_ptr,
        temp_bytes,
        (void* const*)device_decomp_out,
        NULL,
        stream);
    REQUIRE(status == hipcompSuccess);
    
    // Verify correctness
    for (size_t i = 0; i < batch_size; i++) {
      HIP_CHECK(hipMemcpy(
          host_output[i],
          host_decomp_out[i],
          host_batch_bytes[i],
          hipMemcpyDeviceToHost));
      for (size_t j = 0; j < host_batch_bytes[i] / sizeof(T); ++j) {
        REQUIRE(host_output[i][j] == host_input[i][j]);
      }
    }
  }

  hipcompStatus_t* device_statuses;
  HIP_CHECK(hipMalloc(
      (void**)&device_statuses, sizeof(*device_statuses) * batch_size));
  status = decompressAsync(
      (const void* const*)device_comp_out,
      device_comp_out_bytes,
      device_batch_bytes,
      device_decomp_out_bytes,
      batch_size,
      device_temp_ptr,
      temp_bytes,
      (void* const*)device_decomp_out,
      device_statuses,
      stream);
  REQUIRE(status == hipcompSuccess);

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipStreamDestroy(stream));

  // check statuses
  hipcompStatus_t* host_statuses = malloc(sizeof(*device_statuses) * batch_size);
  HIP_CHECK(hipMemcpy(
      host_statuses,
      device_statuses,
      sizeof(*device_statuses) * batch_size,
      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(device_statuses));

  for (size_t i = 0; i < batch_size; ++i) {
    REQUIRE(host_statuses[i] == hipcompSuccess);
  }
  free(host_statuses);

  // check output bytes
  HIP_CHECK(hipMemcpy(
      host_decomp_bytes,
      device_decomp_out_bytes,
      sizeof(*host_decomp_bytes) * batch_size,
      hipMemcpyDeviceToHost));
  for (size_t i = 0; i < batch_size; ++i) {
    REQUIRE(host_decomp_bytes[i] == host_batch_bytes[i]);
  }
  free(host_decomp_bytes);
  HIP_CHECK(hipFree(device_decomp_out_bytes));

  HIP_CHECK(hipFree(device_batch_bytes));
  HIP_CHECK(hipFree(device_comp_out_bytes));
  HIP_CHECK(hipFree(device_temp_ptr));

  for (size_t i = 0; i < batch_size; i++) {
    HIP_CHECK(hipMemcpy(
        host_output[i],
        host_decomp_out[i],
        host_batch_bytes[i],
        hipMemcpyDeviceToHost));
    // Verify correctness
    for (size_t j = 0; j < host_batch_bytes[i] / sizeof(T); ++j) {
      REQUIRE(host_output[i][j] == host_input[i][j]);
    }
    free(host_input[i]);
  }
  free(host_input);
  free(host_batch_bytes);

  for (size_t i = 0; i < batch_size; i++) {
    HIP_CHECK(hipFree(host_comp_out[i]));
    HIP_CHECK(hipFree(host_decomp_out[i]));
    free(host_output[i]);
  }
  HIP_CHECK(hipFree(device_comp_out));
  free(host_output);
  free(host_comp_out);
  free(host_decomp_out);

  return PASS_TEST;
}

int test_generic_batch_decompression_errors(
    const size_t batch_size, const size_t min_size, const size_t max_size)
{
  typedef int T;

  // in this test, we try to decompress random data
  // -- first we try to get the size of it, which should report 0, or a size
  // larger than the input (see NOTE: below).
  // -- then we try to decompress it, which should report an invalid status

  // set a constant seed
  srand(0);

  // prepare input and output on host
  size_t* host_batch_sizes = malloc(batch_size * sizeof(size_t));
  for (size_t i = 0; i < batch_size; ++i) {
    if (max_size > min_size) {
      host_batch_sizes[i] = (rand() % (max_size - min_size)) + min_size;
    } else if (max_size == min_size) {
      host_batch_sizes[i] = max_size;
    } else {
      printf("Invalid max_size (%zu) / min_size (%zu)\n", max_size, min_size);
      return FAIL_TEST;
    }
  }

  size_t* host_batch_bytes = malloc(batch_size * sizeof(size_t));
  size_t max_chunk_size = 0;
  for (size_t i = 0; i < batch_size; ++i) {
    host_batch_bytes[i] = sizeof(T) * host_batch_sizes[i];
    if (host_batch_bytes[i] > max_chunk_size) {
      max_chunk_size = host_batch_bytes[i];
    }
  }

  T** host_input = malloc(sizeof(T*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    host_input[i] = malloc(sizeof(T) * host_batch_sizes[i]);
    for (size_t j = 0; j < host_batch_sizes[i]; ++j) {
      // make sure there should be some repeats to compress
      host_input[i][j] = (rand() % 4) + 300;
    }
  }
  free(host_batch_sizes);

  T** host_output = malloc(sizeof(T*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    host_output[i] = malloc(host_batch_bytes[i]);
  }

  // prepare gpu buffers
  void** host_in_ptrs = malloc(sizeof(void*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    HIP_CHECK(hipMalloc(&host_in_ptrs[i], host_batch_bytes[i]));
    HIP_CHECK(hipMemcpy(
        host_in_ptrs[i],
        host_input[i],
        host_batch_bytes[i],
        hipMemcpyHostToDevice));
  }
  void** device_in_pointers;
  HIP_CHECK(hipMalloc(
      (void**)&device_in_pointers, sizeof(*device_in_pointers) * batch_size));
  HIP_CHECK(hipMemcpy(
      device_in_pointers,
      host_in_ptrs,
      sizeof(*device_in_pointers) * batch_size,
      hipMemcpyHostToDevice));

  size_t* device_batch_bytes;
  HIP_CHECK(hipMalloc(
      (void**)&device_batch_bytes, sizeof(*device_batch_bytes) * batch_size));
  HIP_CHECK(hipMemcpy(
      device_batch_bytes,
      host_batch_bytes,
      sizeof(*device_batch_bytes) * batch_size,
      hipMemcpyHostToDevice));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  hipcompStatus_t status;

  // attempt to get the size
  size_t* device_decomp_out_bytes;
  HIP_CHECK(hipMalloc(
      (void**)&device_decomp_out_bytes,
      sizeof(*device_decomp_out_bytes) * batch_size));
  // initially set all sizes to -1
  HIP_CHECK(hipMemset(
      device_decomp_out_bytes,
      -1,
      sizeof(*device_decomp_out_bytes) * batch_size));

  status = decompressGetSizeAsync(
      (const void* const*)device_in_pointers,
      device_batch_bytes,
      device_decomp_out_bytes,
      batch_size,
      stream);
  REQUIRE(status == hipcompSuccess);
  HIP_CHECK(hipStreamSynchronize(stream));

  // copy the output sizes down and check them
  size_t* host_decomp_bytes = malloc(sizeof(size_t) * batch_size);
  HIP_CHECK(hipMemcpy(
      host_decomp_bytes,
      device_decomp_out_bytes,
      sizeof(*host_decomp_bytes) * batch_size,
      hipMemcpyDeviceToHost));

  // We can't gaurantee that decompressor fails to get a size from the data,
  // so here we can only check that the size has been written to.
  for (size_t i = 0; i < batch_size; ++i) {
    REQUIRE(host_decomp_bytes[i] != (size_t)-1);
  }

  // next set the output buffers to be invalid for the returned sizes
  for (size_t i = 0; i < batch_size; ++i) {
    if (host_decomp_bytes[i] == 0
        || host_decomp_bytes[i] > host_batch_bytes[i]) {
      // We either discovered and invalid chunk when getting the size, or the
      // decompress things will it be decompressed to something larger than
      // `host_batch_bytes[i]`, either way specifying `host_batch_bytes[i]`
      // as the output space should cause it to fail.
      host_decomp_bytes[i] = host_batch_bytes[i];
    } else {
      // We're in danger of this noise successfully decompressing, so we
      // intentionally give it a smaller buffer than it requires.
      host_decomp_bytes[i] = host_decomp_bytes[i] - 1;
    }
  }
  HIP_CHECK(hipMemcpy(
      device_decomp_out_bytes,
      host_decomp_bytes,
      sizeof(*device_decomp_out_bytes) * batch_size,
      hipMemcpyHostToDevice));

  // attempt to decompress
  size_t temp_bytes;
  status = decompressGetTempSize(batch_size, max_chunk_size, &temp_bytes);

  void* device_temp_ptr;
  HIP_CHECK(hipMalloc(&device_temp_ptr, temp_bytes));

  void** host_decomp_out = malloc(sizeof(void*) * batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    HIP_CHECK(hipMalloc(&host_decomp_out[i], host_decomp_bytes[i]));
  }
  void** device_decomp_out;
  hipMalloc(
      (void**)&device_decomp_out, sizeof(*device_decomp_out) * batch_size);
  HIP_CHECK(hipMemcpy(
      device_decomp_out,
      host_decomp_out,
      sizeof(*device_decomp_out) * batch_size,
      hipMemcpyHostToDevice));

  hipcompStatus_t* device_statuses;
  HIP_CHECK(hipMalloc(
      (void**)&device_statuses, sizeof(*device_statuses) * batch_size));
  status = decompressAsync(
      (const void* const*)device_in_pointers,
      device_batch_bytes,
      device_decomp_out_bytes,
      device_decomp_out_bytes,
      batch_size,
      device_temp_ptr,
      temp_bytes,
      (void* const*)device_decomp_out,
      device_statuses,
      stream);
  REQUIRE(status == hipcompSuccess);

  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipStreamDestroy(stream));

  // clean up inputs
  for (size_t i = 0; i < batch_size; ++i) {
    HIP_CHECK(hipFree(host_in_ptrs[i]));
  }
  hipFree(device_in_pointers);
  free(host_in_ptrs);

  // check statuses
  hipcompStatus_t* host_statuses = malloc(sizeof(*device_statuses) * batch_size);
  HIP_CHECK(hipMemcpy(
      host_statuses,
      device_statuses,
      sizeof(*device_statuses) * batch_size,
      hipMemcpyDeviceToHost));
  HIP_CHECK(hipFree(device_statuses));

  for (size_t i = 0; i < batch_size; ++i) {
    if (host_statuses[i] != hipcompErrorCannotDecompress) {
    }
    REQUIRE(host_statuses[i] == hipcompErrorCannotDecompress);
  }
  free(host_statuses);
  HIP_CHECK(hipFree(device_decomp_out_bytes));

  HIP_CHECK(hipFree(device_batch_bytes));
  HIP_CHECK(hipFree(device_temp_ptr));

  for (size_t i = 0; i < batch_size; i++) {
    free(host_input[i]);
  }
  free(host_input);
  free(host_batch_bytes);

  for (size_t i = 0; i < batch_size; i++) {
    HIP_CHECK(hipFree(host_decomp_out[i]));
    free(host_output[i]);
  }
  free(host_output);
  free(host_decomp_out);

  return PASS_TEST;
}

#define TEST(bs, min, max, num_tests, rv, crash_safe, support_nullptr)         \
  do {                                                                         \
    ++(num_tests);                                                             \
    if (!test_generic_batch_compression_and_decompression(bs, min, max, support_nullptr)) {     \
      printf(                                                                  \
          "compression and decompression test failed %dx[%d:%d]\n",            \
          (int)(bs),                                                           \
          (int)(min),                                                          \
          (int)(max));                                                         \
      ++(rv);                                                                  \
    }                                                                          \
    if (crash_safe) {                                                          \
      if (!test_generic_batch_decompression_errors(bs, min, max)) {            \
        printf(                                                                \
            "decompression errors test failed %dx[%d:%d]\n",                   \
            (int)(bs),                                                         \
            (int)(min),                                                        \
            (int)(max));                                                       \
        ++(rv);                                                                \
      }                                                                        \
    }                                                                          \
  } while (0)

int main(int argc, char** argv)
{
  if (argc != 1) {
    printf("ERROR: %s accepts no arguments.\n", argv[0]);
    return 1;
  }

  int num_tests = 0;
  int num_failed_tests = 0;

#ifdef CRASH_SAFE
  const int crash_safe = 1;
#else
  const int crash_safe = 0;
#endif

#ifdef SUPPORT_NULLPTR_APIS
  const int support_nullptr = 1;
#else 
  const int support_nullptr = 0;
#endif

  // these macros count the number of failed tests
  TEST(1, 100, 100, num_tests, num_failed_tests, crash_safe, support_nullptr);
  TEST(1, (1<<16) / sizeof(int), (1<<16) / sizeof(int), num_tests, num_failed_tests, crash_safe, support_nullptr);
  TEST(11, 1000, 10000, num_tests, num_failed_tests, crash_safe, support_nullptr);
  TEST(127, 10000, (1<<16) / sizeof(int), num_tests, num_failed_tests, crash_safe, support_nullptr);
  TEST(1025, 100, (1<<16) / sizeof(int), num_tests, num_failed_tests, crash_safe, support_nullptr);
  TEST(10025, 100, 1000, num_tests, num_failed_tests, crash_safe, support_nullptr);

  if (num_failed_tests == 0) {
    printf(
        "SUCCESS: All tests passed: %d/%d\n",
        (num_tests - num_failed_tests),
        num_tests);
  } else {
    printf("FAILURE: %d/%d tests failed\n", num_failed_tests, num_tests);
  }

  // rely on exit code of 0 being a success, and anything else being a failure
  return num_failed_tests;
}