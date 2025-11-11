
/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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

#define CATCH_CONFIG_MAIN

#include "BatchData.hpp"
#include "catch.hpp"
#include "hipcomp/gzip.h"
#include "zlib.h"

#include <iostream>
#include <string>
#include <vector>

#define nvcompAlignmentRequirements_t hipcompAlignmentRequirements_t
#define nvcompBatchedGzipDecompressAsync hipcompBatchedGzipDecompressAsync
#define nvcompBatchedGzipDecompressDefaultOpts                                 \
  hipcompBatchedGzipDecompressDefaultOpts
#define nvcompBatchedGzipDecompressGetRequiredAlignments                       \
  hipcompBatchedGzipDecompressGetRequiredAlignments
#define nvcompBatchedGzipDecompressGetTempSizeAsync                            \
  hipcompBatchedGzipDecompressGetTempSize
#define nvcompBatchedGzipDecompressOpts_t hipcompBatchedGzipDecompressOpts_t
#define nvcompStatus_t hipcompStatus_t
#define nvcompSuccess hipcompSuccess

static void run_test(const std::vector<std::vector<char>> &data, int algo,
                     int compression_level, size_t warmup_iteration_count,
                     size_t total_iteration_count) {
  assert(!data.empty());
  if (warmup_iteration_count >= total_iteration_count) {
    throw std::runtime_error("ERROR: the total iteration count must be greater "
                             "than the warmup iteration count");
  }

  size_t total_bytes = 0;
  for (const std::vector<char> &part : data) {
    total_bytes += part.size();
  }

  std::cout << "----------" << std::endl;
  std::cout << "files: " << data.size() << std::endl;
  std::cout << "uncompressed (B): " << total_bytes << std::endl;

  const size_t chunk_size = 1 << 16;

  // Build up input batch on CPU
  BatchDataCPU input_data_cpu(data, chunk_size);
  const size_t chunk_count = input_data_cpu.size();
  std::cout << "chunks: " << chunk_count << std::endl;

  // compression

  // Allocate and prepare output/compressed batch
  BatchDataCPU compressed_data_cpu(chunk_size, chunk_count);

  // loop over chunks on the CPU, compressing each one
  for (size_t i = 0; i < chunk_count; ++i) {
    // zlib::deflate
    z_stream zs;
    zs.zalloc = NULL;
    zs.zfree = NULL;
    zs.msg = NULL;
    zs.next_in = (Bytef *)input_data_cpu.ptrs()[i];
    zs.avail_in = static_cast<uInt>(input_data_cpu.sizes()[i]);
    zs.next_out = (Bytef *)compressed_data_cpu.ptrs()[i];
    zs.avail_out = static_cast<uInt>(input_data_cpu.sizes()[i]);
    int strategy = Z_DEFAULT_STRATEGY;
    // 15 | 16 to enable gzip header
    int ret = deflateInit2(&zs, 9, Z_DEFLATED,
                           15 | 16 /* enables GZIP header */, 8, strategy);
    if (ret != Z_OK) {
      throw std::runtime_error("Call to deflateInit2 failed: " +
                               std::to_string(ret));
    }
    if ((ret = deflate(&zs, Z_FINISH)) != Z_STREAM_END) {
      throw std::runtime_error("Gzip operation failed: " + std::to_string(ret));
    }
    if ((ret = deflateEnd(&zs)) != Z_OK) {
      throw std::runtime_error("Call to deflateEnd failed: " +
                               std::to_string(ret));
    }
    // set the actual compressed size
    compressed_data_cpu.sizes()[i] = zs.total_out;
  }

  // compute compression ratio
  size_t *compressed_sizes_host = compressed_data_cpu.sizes();
  size_t comp_bytes = 0;
  for (size_t i = 0; i < chunk_count; ++i)
    comp_bytes += compressed_sizes_host[i];

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  nvcompStatus_t status = nvcompSuccess;
  // // Decompression options
  // nvcompBatchedGzipDecompressOpts_t decompress_opts =
  //     nvcompBatchedGzipDecompressDefaultOpts;

  // // Query decompression alignment requirements
  // nvcompAlignmentRequirements_t decompression_alignment_reqs;
  // nvcompStatus_t status =
  // nvcompBatchedGzipDecompressGetRequiredAlignments(
  //     decompress_opts, &decompression_alignment_reqs);
  // if (status != nvcompSuccess) {
  //   throw std::runtime_error(
  //       "ERROR: nvcompBatchedGzipDecompressGetRequiredAlignments() not "
  //       "successful");
  // }

  // Copy compressed data to GPU
  BatchData compressed_data(compressed_data_cpu, true,
                            hipcompGzipRequiredAlignment);

  // Allocate and build up decompression batch on GPU
  BatchData decomp_data(input_data_cpu, false, hipcompGzipRequiredAlignment);

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // CUDA events to measure decompression time
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // deflate GPU decompression
  size_t decomp_temp_bytes;
  status = nvcompBatchedGzipDecompressGetTempSizeAsync(
      chunk_count, 0 /* max uncompressed size */, &decomp_temp_bytes);
  if (status != nvcompSuccess) {
    throw std::runtime_error(
        "nvcompBatchedGzipDecompressGetTempSizeAsync() failed.");
  }

  void *d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

  size_t *d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, chunk_count * sizeof(size_t)));

  nvcompStatus_t *d_status_ptrs;
  CUDA_CHECK(cudaMalloc(&d_status_ptrs, chunk_count * sizeof(nvcompStatus_t)));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto perform_decompression = [&]() {
    if (nvcompBatchedGzipDecompressAsync(
            compressed_data.ptrs(), compressed_data.sizes(),
            decomp_data.sizes(), d_decomp_sizes, chunk_count, d_decomp_temp,
            decomp_temp_bytes, decomp_data.ptrs(), /*decompress_opts,*/
            d_status_ptrs, stream) != nvcompSuccess) {
      throw std::runtime_error(
          "ERROR: nvcompBatchedGzipDecompressAsync() not successful");
    }
  };

  // Run warm-up decompression
  for (size_t iter = 0; iter < warmup_iteration_count; ++iter) {
    perform_decompression();
  }

  // Re-run decompression to get throughput
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (size_t iter = warmup_iteration_count; iter < total_iteration_count;
       ++iter) {
    perform_decompression();
  }
  CUDA_CHECK(cudaEventRecord(end, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Validate decompressed data against input
  if (!(input_data_cpu == decomp_data)) {
    throw std::runtime_error("Failed to validate decompressed data");
  } else {
    std::cout << "decompression validated :)" << std::endl;
  }

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
  ms /= total_iteration_count - warmup_iteration_count;

  double decompression_throughput = ((double)total_bytes / ms) * 1e-6;
  std::cout << "decompression throughput (GB/s): " << decompression_throughput
            << std::endl;

  CUDA_CHECK(cudaFree(d_decomp_temp));
  CUDA_CHECK(cudaFree(d_decomp_sizes));
  CUDA_CHECK(cudaFree(d_status_ptrs));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST_CASE("decomp Gzip deflate", "[deflate, hipcomp]") {
  auto data = generate_multi_test_sequence_repeat_62(300);

  int algo = 0;
  int compression_level = 6;
  size_t warmup_iteration_count = 2;
  size_t total_iteration_count = 5;

  // 0 libdeflate, 1 zlib_compress2, 2 zlib_deflate
  run_test(data, algo, compression_level, warmup_iteration_count,
           total_iteration_count);
}

TEST_CASE("decomp Gzip zlib_compress2", "[zlib_compress2, hipcomp]") {
  auto data = generate_multi_test_sequence_repeat_62(300);

  int algo = 1;
  int compression_level = 6;
  size_t warmup_iteration_count = 2;
  size_t total_iteration_count = 5;

  // 0 libdeflate, 1 zlib_compress2, 2 zlib_deflate
  run_test(data, algo, compression_level, warmup_iteration_count,
           total_iteration_count);
}

TEST_CASE("decomp Gzip zlib_deflate", "[zlib_deflate, hipcomp]") {
  auto data = generate_multi_test_sequence_repeat_62(300);

  int algo = 2;
  int compression_level = 6;
  size_t warmup_iteration_count = 2;
  size_t total_iteration_count = 5;

  // 0 libdeflate, 1 zlib_compress2, 2 zlib_deflate
  run_test(data, algo, compression_level, warmup_iteration_count,
           total_iteration_count);
}
