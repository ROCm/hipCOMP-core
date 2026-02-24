
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
// Modifications Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights
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
#if defined(CUDA_BACKEND)
#include "nvcomp/zstd.h"
#else
#include "hipcomp/zstd.h"
#endif
#include "zstd.h"

#include <iostream>
#include <string>
#include <vector>

static void run_test(const std::vector<std::vector<char>> &data,
                     int compression_level, size_t warmup_iteration_count,
                     size_t total_iteration_count,
                     size_t chunk_size = 1 << 16) {
  assert(!data.empty());
  if (warmup_iteration_count >= total_iteration_count) {
    throw std::runtime_error("ERROR: the total iteration count must be greater "
                             "than the warmup iteration count");
  }

  size_t total_bytes = 0;
  for (const std::vector<char> &part : data) {
    total_bytes += part.size();
  }

  std::cout << "files: " << data.size() << std::endl;

  // Build up input batch on CPU
  BatchDataCPU input_data_cpu(data, chunk_size);
  const size_t chunk_count = input_data_cpu.size();

  // compression

  // Allocate and prepare output/compressed batch
  BatchDataCPU compressed_data_cpu(ZSTD_compressBound(chunk_size), chunk_count);

  // loop over chunks on the CPU, compressing each one
  const auto min_compression_level = ZSTD_minCLevel();
  const auto max_compression_level = ZSTD_maxCLevel();
  if (compression_level < min_compression_level ||
      compression_level > max_compression_level) {
    throw std::runtime_error(
        "Unsupported compression level: " + std::to_string(compression_level) +
        ". Supported range: " + std::to_string(min_compression_level) + " - " +
        std::to_string(max_compression_level));
  }
  for (size_t i = 0; i < chunk_count; ++i) {
    size_t size = ZSTD_compress(
        compressed_data_cpu.ptrs()[i], compressed_data_cpu.sizes()[i],
        input_data_cpu.ptrs()[i], input_data_cpu.sizes()[i], compression_level);
    if (ZSTD_isError(size)) {
      throw std::runtime_error("Zstandard CPU failed to compress chunk " +
                               std::to_string(i) +
                               ". Error code: " + std::to_string(size) +
                               ", Message: " + ZSTD_getErrorName(size));
    }
    compressed_data_cpu.sizes()[i] = size;
  }

  // compute compression ratio
  size_t comp_bytes =
      std::accumulate(compressed_data_cpu.sizes(),
                      compressed_data_cpu.sizes() + chunk_count, size_t(0));

  std::cout << "comp_size: " << comp_bytes
            << ", compressed ratio: " << std::fixed << std::setprecision(2)
            << (double)total_bytes / comp_bytes << std::endl;

  nvcompStatus_t status = nvcompSuccess;
#ifdef CUDA_BACKEND
  // Decompression options
  nvcompBatchedZstdDecompressOpts_t decompress_opts =
      nvcompBatchedZstdDecompressDefaultOpts;

  // Query decompression alignment requirements
  nvcompAlignmentRequirements_t decompression_alignment_reqs_s;
  status = nvcompBatchedZstdDecompressGetRequiredAlignments(
      decompress_opts, &decompression_alignment_reqs_s);
  if (status != nvcompSuccess) {
    throw std::runtime_error(
        "ERROR: nvcompBatchedZstdDecompressGetRequiredAlignments() not "
        "successful");
  }
  auto decompression_alignment_reqs_in = decompression_alignment_reqs_s.input;
  auto decompression_alignment_reqs_out = decompression_alignment_reqs_s.output;
#else
  auto decompression_alignment_reqs_in = hipcompZstdRequiredAlignment;
  auto decompression_alignment_reqs_out = hipcompZstdRequiredAlignment;
#endif

  // Copy compressed data to GPU
  BatchData compressed_data(compressed_data_cpu, true,
                            decompression_alignment_reqs_in);

  // Allocate and build up decompression batch on GPU
  BatchData decomp_data(input_data_cpu, false,
                        decompression_alignment_reqs_out);

  // Create CUDA stream
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // Query actual required decompression sizes
  size_t *d_queried_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_queried_decomp_sizes, chunk_count * sizeof(size_t)));

  status = hipcompBatchedZstdGetDecompressSizeAsync(
      compressed_data.ptrs(), compressed_data.sizes(), d_queried_decomp_sizes,
      chunk_count, stream);
  if (status != nvcompSuccess) {
    throw std::runtime_error(
        "hipcompBatchedZstdGetDecompressSizeAsync() failed.");
  }

  // Copy queried sizes back to host for validation
  std::vector<size_t> queried_decomp_sizes(chunk_count);
  CUDA_CHECK(cudaMemcpyAsync(
      queried_decomp_sizes.data(), d_queried_decomp_sizes,
      chunk_count * sizeof(size_t), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Compare queried sizes with allocated buffer sizes and print information
  std::cout << "Decompression buffer size validation:" << std::endl;
  bool size_warning = false;
  size_t total_queried_size = 0;
  size_t total_allocated_size = 0;
  for (size_t i = 0; i < chunk_count; ++i) {
    total_queried_size += queried_decomp_sizes[i];
    total_allocated_size += decomp_data.sizes()[i];
    if (queried_decomp_sizes[i] > decomp_data.sizes()[i]) {
      std::cout << "  WARNING: Chunk " << i << " requires "
                << queried_decomp_sizes[i] << " bytes but only "
                << decomp_data.sizes()[i] << " bytes allocated!" << std::endl;
      size_warning = true;
    }
  }
  std::cout << "  Total queried size: " << total_queried_size << " bytes"
            << std::endl;
  std::cout << "  Total allocated size: " << total_allocated_size << " bytes"
            << std::endl;
  if (size_warning) {
    std::cout << "  WARNING: Some output buffers may be undersized!"
              << std::endl;
  } else {
    std::cout << "  All output buffers are adequately sized." << std::endl;
  }

  CUDA_CHECK(cudaFree(d_queried_decomp_sizes));

  // CUDA events to measure decompression time
  cudaEvent_t start, end;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // Zstandard GPU decompression
  // Determine scratch space needed asynchronously
  size_t decomp_temp_bytes_async;
  status = nvcompBatchedZstdDecompressGetTempSizeAsync(
#ifdef CUDA_BACKEND
      chunk_count, chunk_size, decompress_opts, &decomp_temp_bytes_async,
      chunk_count * chunk_size
#else
      chunk_count, 0 /* max uncompressed size */, &decomp_temp_bytes_async
#endif
  );
  if (status != nvcompSuccess) {
    throw std::runtime_error(
        "nvcompBatchedZstdDecompressGetTempSizeAsync() failed.");
  }

#ifdef CUDA_BACKEND // TODO(HIP/AMD): Not implemented
  // Determine scratch space needed synchronously
  size_t decomp_temp_bytes_sync;
  status = nvcompBatchedZstdDecompressGetTempSizeSync(
      compressed_data.ptrs(), compressed_data.sizes(), chunk_count, chunk_size,
      &decomp_temp_bytes_sync, chunk_size * chunk_count,
#ifdef CUDA_BACKEND
      decompress_opts,
#endif
      d_status_ptrs, stream);
  if (status != nvcompSuccess) {
    throw std::runtime_error(
        "nvcompBatchedZstdDecompressGetTempSizeSync() failed.");
  }
  size_t decomp_temp_bytes =
      std::min(decomp_temp_bytes_sync, decomp_temp_bytes_async);
#else
  size_t decomp_temp_bytes = decomp_temp_bytes_async;
#endif

  void *d_decomp_temp;
  CUDA_CHECK(cudaMalloc(&d_decomp_temp, decomp_temp_bytes));

  size_t *d_decomp_sizes;
  CUDA_CHECK(cudaMalloc(&d_decomp_sizes, chunk_count * sizeof(size_t)));

  nvcompStatus_t *d_status_ptrs;
  CUDA_CHECK(cudaMalloc(&d_status_ptrs, chunk_count * sizeof(nvcompStatus_t)));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto perform_decompression = [&]() {
    if (nvcompBatchedZstdDecompressAsync(
            compressed_data.ptrs(), compressed_data.sizes(),
            decomp_data.sizes(), d_decomp_sizes, chunk_count, d_decomp_temp,
            decomp_temp_bytes, decomp_data.ptrs(),
#ifdef CUDA_BACKEND
            decompress_opts,
#endif
            d_status_ptrs, stream) != nvcompSuccess) {
      throw std::runtime_error(
          "ERROR: nvcompBatchedZstdDecompressAsync() not successful");
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

TEST_CASE("decomp Zstd zstd", "[zstd, hipcomp]") {
  auto data = generate_multi_test_sequence_repeat_62(300);

  int compression_level = 6;
  size_t warmup_iteration_count = 2;
  size_t total_iteration_count = 5;

  run_test(data, compression_level, warmup_iteration_count,
           total_iteration_count);
}

#define DEFINE_ALIAS(NAME, SCENARIO, CHUNK, DATA)                              \
  using NAME = FullMatrixTag<SCENARIO, CHUNK, DATA>;

#define DEFINE_ALL_DATA_SIZES(PREFIX, SCENARIO, CHUNK)                         \
  DEFINE_ALIAS(PREFIX##_TinyInput, SCENARIO, CHUNK, DataSizeClass::Tiny)       \
  DEFINE_ALIAS(PREFIX##_SmallInput, SCENARIO, CHUNK, DataSizeClass::Small)     \
  DEFINE_ALIAS(PREFIX##_MediumInput, SCENARIO, CHUNK, DataSizeClass::Medium)   \
  DEFINE_ALIAS(PREFIX##_LargeInput, SCENARIO, CHUNK, DataSizeClass::Large)     \
  DEFINE_ALIAS(PREFIX##_HugeInput, SCENARIO, CHUNK, DataSizeClass::Huge)

#define DEFINE_ALL_CHUNKS(PREFIX, SCENARIO)                                    \
  DEFINE_ALL_DATA_SIZES(PREFIX##_TinyChunk, SCENARIO, ChunkClass::Tiny)        \
  DEFINE_ALL_DATA_SIZES(PREFIX##_SmallChunk, SCENARIO, ChunkClass::Small)      \
  DEFINE_ALL_DATA_SIZES(PREFIX##_MediumChunk, SCENARIO, ChunkClass::Medium)    \
  DEFINE_ALL_DATA_SIZES(PREFIX##_LargeChunk, SCENARIO, ChunkClass::Large)      \
  DEFINE_ALL_DATA_SIZES(PREFIX##_HugeChunk, SCENARIO, ChunkClass::Huge)

DEFINE_ALL_CHUNKS(LZ77_SingleByte_Long, ZstdTestScenario::LZ77_SingleByte_Long)

DEFINE_ALL_CHUNKS(LZ77_ShortPattern, ZstdTestScenario::LZ77_ShortPattern)

DEFINE_ALL_CHUNKS(LZ77_LargeOffset, ZstdTestScenario::LZ77_LargeOffset)

DEFINE_ALL_CHUNKS(LZ77_WindowBoundary, ZstdTestScenario::LZ77_WindowBoundary)

DEFINE_ALL_CHUNKS(Huffman_Skewed_90_10, ZstdTestScenario::Huffman_Skewed_90_10)

DEFINE_ALL_CHUNKS(Huffman_Skewed_99_1, ZstdTestScenario::Huffman_Skewed_99_1)

DEFINE_ALL_CHUNKS(Huffman_Uniform_256, ZstdTestScenario::Huffman_Uniform_256)

DEFINE_ALL_CHUNKS(Huffman_Bimodal, ZstdTestScenario::Huffman_Bimodal)

DEFINE_ALL_CHUNKS(FSE_ConstantSequences,
                  ZstdTestScenario::FSE_ConstantSequences)

DEFINE_ALL_CHUNKS(FSE_SmallAlphabetSequences,
                  ZstdTestScenario::FSE_SmallAlphabetSequences)

DEFINE_ALL_CHUNKS(FSE_VariableSequences,
                  ZstdTestScenario::FSE_VariableSequences)

DEFINE_ALL_CHUNKS(FSE_PathologicalSequences,
                  ZstdTestScenario::FSE_PathologicalSequences)

DEFINE_ALL_CHUNKS(Mixed_TextLike, ZstdTestScenario::Mixed_TextLike)

DEFINE_ALL_CHUNKS(Mixed_BinaryLike, ZstdTestScenario::Mixed_BinaryLike)

DEFINE_ALL_CHUNKS(Mixed_Bursty, ZstdTestScenario::Mixed_Bursty)

DEFINE_ALL_CHUNKS(Random_Fully, ZstdTestScenario::Random_Fully)

DEFINE_ALL_CHUNKS(Random_Blockwise, ZstdTestScenario::Random_Blockwise)

DEFINE_ALL_CHUNKS(Entropy_WorstCase, ZstdTestScenario::Entropy_WorstCase)

TEMPLATE_TEST_CASE(
    "decomp Zstd full mechanism x chunk x dataset matrix",
    "[zstd][nvcomp][matrix][mechanism]",

    LZ77_SingleByte_Long_TinyChunk_TinyInput,
    LZ77_SingleByte_Long_TinyChunk_SmallInput,
    LZ77_SingleByte_Long_TinyChunk_MediumInput,
    LZ77_SingleByte_Long_TinyChunk_LargeInput,
    LZ77_SingleByte_Long_TinyChunk_HugeInput,
    LZ77_SingleByte_Long_SmallChunk_TinyInput,
    LZ77_SingleByte_Long_SmallChunk_SmallInput,
    LZ77_SingleByte_Long_SmallChunk_MediumInput,
    LZ77_SingleByte_Long_SmallChunk_LargeInput,
    LZ77_SingleByte_Long_SmallChunk_HugeInput,
    LZ77_SingleByte_Long_MediumChunk_TinyInput,
    LZ77_SingleByte_Long_MediumChunk_SmallInput,
    LZ77_SingleByte_Long_MediumChunk_MediumInput,
    LZ77_SingleByte_Long_MediumChunk_LargeInput,
    LZ77_SingleByte_Long_MediumChunk_HugeInput,
    LZ77_SingleByte_Long_LargeChunk_TinyInput,
    LZ77_SingleByte_Long_LargeChunk_SmallInput,
    LZ77_SingleByte_Long_LargeChunk_MediumInput,
    LZ77_SingleByte_Long_LargeChunk_LargeInput,
    LZ77_SingleByte_Long_LargeChunk_HugeInput,
    LZ77_SingleByte_Long_HugeChunk_TinyInput,
    LZ77_SingleByte_Long_HugeChunk_SmallInput,
    LZ77_SingleByte_Long_HugeChunk_MediumInput,
    LZ77_SingleByte_Long_HugeChunk_LargeInput,
    LZ77_SingleByte_Long_HugeChunk_HugeInput,

    LZ77_ShortPattern_TinyChunk_TinyInput,
    LZ77_ShortPattern_TinyChunk_SmallInput,
    LZ77_ShortPattern_TinyChunk_MediumInput,
    LZ77_ShortPattern_TinyChunk_LargeInput,
    LZ77_ShortPattern_TinyChunk_HugeInput,
    LZ77_ShortPattern_SmallChunk_TinyInput,
    LZ77_ShortPattern_SmallChunk_SmallInput,
    LZ77_ShortPattern_SmallChunk_MediumInput,
    LZ77_ShortPattern_SmallChunk_LargeInput,
    LZ77_ShortPattern_SmallChunk_HugeInput,
    LZ77_ShortPattern_MediumChunk_TinyInput,
    LZ77_ShortPattern_MediumChunk_SmallInput,
    LZ77_ShortPattern_MediumChunk_MediumInput,
    LZ77_ShortPattern_MediumChunk_LargeInput,
    LZ77_ShortPattern_MediumChunk_HugeInput,
    LZ77_ShortPattern_LargeChunk_TinyInput,
    LZ77_ShortPattern_LargeChunk_SmallInput,
    LZ77_ShortPattern_LargeChunk_MediumInput,
    LZ77_ShortPattern_LargeChunk_LargeInput,
    LZ77_ShortPattern_LargeChunk_HugeInput,
    LZ77_ShortPattern_HugeChunk_TinyInput,
    LZ77_ShortPattern_HugeChunk_SmallInput,
    LZ77_ShortPattern_HugeChunk_MediumInput,
    LZ77_ShortPattern_HugeChunk_LargeInput,
    LZ77_ShortPattern_HugeChunk_HugeInput,

    LZ77_LargeOffset_TinyChunk_TinyInput, LZ77_LargeOffset_TinyChunk_SmallInput,
    LZ77_LargeOffset_TinyChunk_MediumInput,
    LZ77_LargeOffset_TinyChunk_LargeInput, LZ77_LargeOffset_TinyChunk_HugeInput,
    LZ77_LargeOffset_SmallChunk_TinyInput,
    LZ77_LargeOffset_SmallChunk_SmallInput,
    LZ77_LargeOffset_SmallChunk_MediumInput,
    LZ77_LargeOffset_SmallChunk_LargeInput,
    LZ77_LargeOffset_SmallChunk_HugeInput,
    LZ77_LargeOffset_MediumChunk_TinyInput,
    LZ77_LargeOffset_MediumChunk_SmallInput,
    LZ77_LargeOffset_MediumChunk_MediumInput,
    LZ77_LargeOffset_MediumChunk_LargeInput,
    LZ77_LargeOffset_MediumChunk_HugeInput,
    LZ77_LargeOffset_LargeChunk_TinyInput,
    LZ77_LargeOffset_LargeChunk_SmallInput,
    LZ77_LargeOffset_LargeChunk_MediumInput,
    LZ77_LargeOffset_LargeChunk_LargeInput,
    LZ77_LargeOffset_LargeChunk_HugeInput, LZ77_LargeOffset_HugeChunk_TinyInput,
    LZ77_LargeOffset_HugeChunk_SmallInput,
    LZ77_LargeOffset_HugeChunk_MediumInput,
    LZ77_LargeOffset_HugeChunk_LargeInput, LZ77_LargeOffset_HugeChunk_HugeInput,

    LZ77_WindowBoundary_TinyChunk_TinyInput,
    LZ77_WindowBoundary_TinyChunk_SmallInput,
    LZ77_WindowBoundary_TinyChunk_MediumInput,
    LZ77_WindowBoundary_TinyChunk_LargeInput,
    LZ77_WindowBoundary_TinyChunk_HugeInput,
    LZ77_WindowBoundary_SmallChunk_TinyInput,
    LZ77_WindowBoundary_SmallChunk_SmallInput,
    LZ77_WindowBoundary_SmallChunk_MediumInput,
    LZ77_WindowBoundary_SmallChunk_LargeInput,
    LZ77_WindowBoundary_SmallChunk_HugeInput,
    LZ77_WindowBoundary_MediumChunk_TinyInput,
    LZ77_WindowBoundary_MediumChunk_SmallInput,
    LZ77_WindowBoundary_MediumChunk_MediumInput,
    LZ77_WindowBoundary_MediumChunk_LargeInput,
    LZ77_WindowBoundary_MediumChunk_HugeInput,
    LZ77_WindowBoundary_LargeChunk_TinyInput,
    LZ77_WindowBoundary_LargeChunk_SmallInput,
    LZ77_WindowBoundary_LargeChunk_MediumInput,
    LZ77_WindowBoundary_LargeChunk_LargeInput,
    LZ77_WindowBoundary_LargeChunk_HugeInput,
    LZ77_WindowBoundary_HugeChunk_TinyInput,
    LZ77_WindowBoundary_HugeChunk_SmallInput,
    LZ77_WindowBoundary_HugeChunk_MediumInput,
    LZ77_WindowBoundary_HugeChunk_LargeInput,
    LZ77_WindowBoundary_HugeChunk_HugeInput) {
  using T = TestType;

  constexpr int compression_level = 6;
  constexpr size_t warmup_iteration_count = 2;
  constexpr size_t total_iteration_count = 5;

  std::cout << "\n=====================================\n";
  std::cout << "Scenario : " << scenario_to_string(T::scenario) << "\n";
  std::cout << "Chunking : " << chunk_class_to_string(T::chunk_class) << "\n";
  std::cout << "Dataset  : " << data_size_class_to_string(T::data_size) << "\n";
  std::cout << "=====================================\n";

  const size_t chunk_size = chunk_size_from_class(T::chunk_class);

  auto data = generate_multi_test(T::scenario, T::data_size, chunk_size / 4);

  run_test(data, compression_level, warmup_iteration_count,
           total_iteration_count, chunk_size);
}

TEMPLATE_TEST_CASE(
    "decomp Zstd full mechanism x chunk x dataset matrix",
    "[zstd][nvcomp][matrix][mechanism]",
    Huffman_Skewed_90_10_TinyChunk_TinyInput,
    Huffman_Skewed_90_10_TinyChunk_SmallInput,
    Huffman_Skewed_90_10_TinyChunk_MediumInput,
    Huffman_Skewed_90_10_TinyChunk_LargeInput,
    Huffman_Skewed_90_10_TinyChunk_HugeInput,
    Huffman_Skewed_90_10_SmallChunk_TinyInput,
    Huffman_Skewed_90_10_SmallChunk_SmallInput,
    Huffman_Skewed_90_10_SmallChunk_MediumInput,
    Huffman_Skewed_90_10_SmallChunk_LargeInput,
    Huffman_Skewed_90_10_SmallChunk_HugeInput,
    Huffman_Skewed_90_10_MediumChunk_TinyInput,
    Huffman_Skewed_90_10_MediumChunk_SmallInput,
    Huffman_Skewed_90_10_MediumChunk_MediumInput,
    Huffman_Skewed_90_10_MediumChunk_LargeInput,
    Huffman_Skewed_90_10_MediumChunk_HugeInput,
    Huffman_Skewed_90_10_LargeChunk_TinyInput,
    Huffman_Skewed_90_10_LargeChunk_SmallInput,
    Huffman_Skewed_90_10_LargeChunk_MediumInput,
    Huffman_Skewed_90_10_LargeChunk_LargeInput,
    Huffman_Skewed_90_10_LargeChunk_HugeInput,
    Huffman_Skewed_90_10_HugeChunk_TinyInput,
    Huffman_Skewed_90_10_HugeChunk_SmallInput,
    Huffman_Skewed_90_10_HugeChunk_MediumInput,
    Huffman_Skewed_90_10_HugeChunk_LargeInput,
    Huffman_Skewed_90_10_HugeChunk_HugeInput,

    Huffman_Skewed_99_1_TinyChunk_TinyInput,
    Huffman_Skewed_99_1_TinyChunk_SmallInput,
    Huffman_Skewed_99_1_TinyChunk_MediumInput,
    Huffman_Skewed_99_1_TinyChunk_LargeInput,
    Huffman_Skewed_99_1_TinyChunk_HugeInput,
    Huffman_Skewed_99_1_SmallChunk_TinyInput,
    Huffman_Skewed_99_1_SmallChunk_SmallInput,
    Huffman_Skewed_99_1_SmallChunk_MediumInput,
    Huffman_Skewed_99_1_SmallChunk_LargeInput,
    Huffman_Skewed_99_1_SmallChunk_HugeInput,
    Huffman_Skewed_99_1_MediumChunk_TinyInput,
    Huffman_Skewed_99_1_MediumChunk_SmallInput,
    Huffman_Skewed_99_1_MediumChunk_MediumInput,
    Huffman_Skewed_99_1_MediumChunk_LargeInput,
    Huffman_Skewed_99_1_MediumChunk_HugeInput,
    Huffman_Skewed_99_1_LargeChunk_TinyInput,
    Huffman_Skewed_99_1_LargeChunk_SmallInput,
    Huffman_Skewed_99_1_LargeChunk_MediumInput,
    Huffman_Skewed_99_1_LargeChunk_LargeInput,
    Huffman_Skewed_99_1_LargeChunk_HugeInput,
    Huffman_Skewed_99_1_HugeChunk_TinyInput,
    Huffman_Skewed_99_1_HugeChunk_SmallInput,
    Huffman_Skewed_99_1_HugeChunk_MediumInput,
    Huffman_Skewed_99_1_HugeChunk_LargeInput,
    Huffman_Skewed_99_1_HugeChunk_HugeInput,

    Huffman_Uniform_256_TinyChunk_TinyInput,
    Huffman_Uniform_256_TinyChunk_SmallInput,
    Huffman_Uniform_256_TinyChunk_MediumInput,
    Huffman_Uniform_256_TinyChunk_LargeInput,
    Huffman_Uniform_256_TinyChunk_HugeInput,
    Huffman_Uniform_256_SmallChunk_TinyInput,
    Huffman_Uniform_256_SmallChunk_SmallInput,
    Huffman_Uniform_256_SmallChunk_MediumInput,
    Huffman_Uniform_256_SmallChunk_LargeInput,
    Huffman_Uniform_256_SmallChunk_HugeInput,
    Huffman_Uniform_256_MediumChunk_TinyInput,
    Huffman_Uniform_256_MediumChunk_SmallInput,
    Huffman_Uniform_256_MediumChunk_MediumInput,
    Huffman_Uniform_256_MediumChunk_LargeInput,
    Huffman_Uniform_256_MediumChunk_HugeInput,
    Huffman_Uniform_256_LargeChunk_TinyInput,
    Huffman_Uniform_256_LargeChunk_SmallInput,
    Huffman_Uniform_256_LargeChunk_MediumInput,
    Huffman_Uniform_256_LargeChunk_LargeInput,
    Huffman_Uniform_256_LargeChunk_HugeInput,
    Huffman_Uniform_256_HugeChunk_TinyInput,
    Huffman_Uniform_256_HugeChunk_SmallInput,
    Huffman_Uniform_256_HugeChunk_MediumInput,
    Huffman_Uniform_256_HugeChunk_LargeInput,
    Huffman_Uniform_256_HugeChunk_HugeInput,

    Huffman_Bimodal_TinyChunk_TinyInput, Huffman_Bimodal_TinyChunk_SmallInput,
    Huffman_Bimodal_TinyChunk_MediumInput, Huffman_Bimodal_TinyChunk_LargeInput,
    Huffman_Bimodal_TinyChunk_HugeInput, Huffman_Bimodal_SmallChunk_TinyInput,
    Huffman_Bimodal_SmallChunk_SmallInput,
    Huffman_Bimodal_SmallChunk_MediumInput,
    Huffman_Bimodal_SmallChunk_LargeInput, Huffman_Bimodal_SmallChunk_HugeInput,
    Huffman_Bimodal_MediumChunk_TinyInput,
    Huffman_Bimodal_MediumChunk_SmallInput,
    Huffman_Bimodal_MediumChunk_MediumInput,
    Huffman_Bimodal_MediumChunk_LargeInput,
    Huffman_Bimodal_MediumChunk_HugeInput, Huffman_Bimodal_LargeChunk_TinyInput,
    Huffman_Bimodal_LargeChunk_SmallInput,
    Huffman_Bimodal_LargeChunk_MediumInput,
    Huffman_Bimodal_LargeChunk_LargeInput, Huffman_Bimodal_LargeChunk_HugeInput,
    Huffman_Bimodal_HugeChunk_TinyInput, Huffman_Bimodal_HugeChunk_SmallInput,
    Huffman_Bimodal_HugeChunk_MediumInput, Huffman_Bimodal_HugeChunk_LargeInput,
    Huffman_Bimodal_HugeChunk_HugeInput) {
  using T = TestType;

  constexpr int compression_level = 6;
  constexpr size_t warmup_iteration_count = 2;
  constexpr size_t total_iteration_count = 5;

  std::cout << "\n=====================================\n";
  std::cout << "Scenario : " << scenario_to_string(T::scenario) << "\n";
  std::cout << "Chunking : " << chunk_class_to_string(T::chunk_class) << "\n";
  std::cout << "Dataset  : " << data_size_class_to_string(T::data_size) << "\n";
  std::cout << "=====================================\n";

  const size_t chunk_size = chunk_size_from_class(T::chunk_class);

  auto data = generate_multi_test(T::scenario, T::data_size, chunk_size / 4);

  run_test(data, compression_level, warmup_iteration_count,
           total_iteration_count, chunk_size);
}

TEMPLATE_TEST_CASE("decomp Zstd full mechanism x chunk x dataset matrix",
                   "[zstd][nvcomp][matrix][mechanism]",
                   FSE_ConstantSequences_TinyChunk_TinyInput,
                   FSE_ConstantSequences_TinyChunk_SmallInput,
                   FSE_ConstantSequences_TinyChunk_MediumInput,
                   FSE_ConstantSequences_TinyChunk_LargeInput,
                   FSE_ConstantSequences_TinyChunk_HugeInput,
                   FSE_ConstantSequences_SmallChunk_TinyInput,
                   FSE_ConstantSequences_SmallChunk_SmallInput,
                   FSE_ConstantSequences_SmallChunk_MediumInput,
                   FSE_ConstantSequences_SmallChunk_LargeInput,
                   FSE_ConstantSequences_SmallChunk_HugeInput,
                   FSE_ConstantSequences_MediumChunk_TinyInput,
                   FSE_ConstantSequences_MediumChunk_SmallInput,
                   FSE_ConstantSequences_MediumChunk_MediumInput,
                   FSE_ConstantSequences_MediumChunk_LargeInput,
                   FSE_ConstantSequences_MediumChunk_HugeInput,
                   FSE_ConstantSequences_LargeChunk_TinyInput,
                   FSE_ConstantSequences_LargeChunk_SmallInput,
                   FSE_ConstantSequences_LargeChunk_MediumInput,
                   FSE_ConstantSequences_LargeChunk_LargeInput,
                   FSE_ConstantSequences_LargeChunk_HugeInput,
                   FSE_ConstantSequences_HugeChunk_TinyInput,
                   FSE_ConstantSequences_HugeChunk_SmallInput,
                   FSE_ConstantSequences_HugeChunk_MediumInput,
                   FSE_ConstantSequences_HugeChunk_LargeInput,
                   FSE_ConstantSequences_HugeChunk_HugeInput,

                   FSE_SmallAlphabetSequences_TinyChunk_TinyInput,
                   FSE_SmallAlphabetSequences_TinyChunk_SmallInput,
                   FSE_SmallAlphabetSequences_TinyChunk_MediumInput,
                   FSE_SmallAlphabetSequences_TinyChunk_LargeInput,
                   FSE_SmallAlphabetSequences_TinyChunk_HugeInput,
                   FSE_SmallAlphabetSequences_SmallChunk_TinyInput,
                   FSE_SmallAlphabetSequences_SmallChunk_SmallInput,
                   FSE_SmallAlphabetSequences_SmallChunk_MediumInput,
                   FSE_SmallAlphabetSequences_SmallChunk_LargeInput,
                   FSE_SmallAlphabetSequences_SmallChunk_HugeInput,
                   FSE_SmallAlphabetSequences_MediumChunk_TinyInput,
                   FSE_SmallAlphabetSequences_MediumChunk_SmallInput,
                   FSE_SmallAlphabetSequences_MediumChunk_MediumInput,
                   FSE_SmallAlphabetSequences_MediumChunk_LargeInput,
                   FSE_SmallAlphabetSequences_MediumChunk_HugeInput,
                   FSE_SmallAlphabetSequences_LargeChunk_TinyInput,
                   FSE_SmallAlphabetSequences_LargeChunk_SmallInput,
                   FSE_SmallAlphabetSequences_LargeChunk_MediumInput,
                   FSE_SmallAlphabetSequences_LargeChunk_LargeInput,
                   FSE_SmallAlphabetSequences_LargeChunk_HugeInput,
                   FSE_SmallAlphabetSequences_HugeChunk_TinyInput,
                   FSE_SmallAlphabetSequences_HugeChunk_SmallInput,
                   FSE_SmallAlphabetSequences_HugeChunk_MediumInput,
                   FSE_SmallAlphabetSequences_HugeChunk_LargeInput,
                   FSE_SmallAlphabetSequences_HugeChunk_HugeInput,

                   FSE_VariableSequences_TinyChunk_TinyInput,
                   FSE_VariableSequences_TinyChunk_SmallInput,
                   FSE_VariableSequences_TinyChunk_MediumInput,
                   FSE_VariableSequences_TinyChunk_LargeInput,
                   FSE_VariableSequences_TinyChunk_HugeInput,
                   FSE_VariableSequences_SmallChunk_TinyInput,
                   FSE_VariableSequences_SmallChunk_SmallInput,
                   FSE_VariableSequences_SmallChunk_MediumInput,
                   FSE_VariableSequences_SmallChunk_LargeInput,
                   FSE_VariableSequences_SmallChunk_HugeInput,
                   FSE_VariableSequences_MediumChunk_TinyInput,
                   FSE_VariableSequences_MediumChunk_SmallInput,
                   FSE_VariableSequences_MediumChunk_MediumInput,
                   FSE_VariableSequences_MediumChunk_LargeInput,
                   FSE_VariableSequences_MediumChunk_HugeInput,
                   FSE_VariableSequences_LargeChunk_TinyInput,
                   FSE_VariableSequences_LargeChunk_SmallInput,
                   FSE_VariableSequences_LargeChunk_MediumInput,
                   FSE_VariableSequences_LargeChunk_LargeInput,
                   FSE_VariableSequences_LargeChunk_HugeInput,
                   FSE_VariableSequences_HugeChunk_TinyInput,
                   FSE_VariableSequences_HugeChunk_SmallInput,
                   FSE_VariableSequences_HugeChunk_MediumInput,
                   FSE_VariableSequences_HugeChunk_LargeInput,
                   FSE_VariableSequences_HugeChunk_HugeInput,

                   FSE_PathologicalSequences_TinyChunk_TinyInput,
                   FSE_PathologicalSequences_TinyChunk_SmallInput,
                   FSE_PathologicalSequences_TinyChunk_MediumInput,
                   FSE_PathologicalSequences_TinyChunk_LargeInput,
                   FSE_PathologicalSequences_TinyChunk_HugeInput,
                   FSE_PathologicalSequences_SmallChunk_TinyInput,
                   FSE_PathologicalSequences_SmallChunk_SmallInput,
                   FSE_PathologicalSequences_SmallChunk_MediumInput,
                   FSE_PathologicalSequences_SmallChunk_LargeInput,
                   FSE_PathologicalSequences_SmallChunk_HugeInput,
                   FSE_PathologicalSequences_MediumChunk_TinyInput,
                   FSE_PathologicalSequences_MediumChunk_SmallInput,
                   FSE_PathologicalSequences_MediumChunk_MediumInput,
                   FSE_PathologicalSequences_MediumChunk_LargeInput,
                   FSE_PathologicalSequences_MediumChunk_HugeInput,
                   FSE_PathologicalSequences_LargeChunk_TinyInput,
                   FSE_PathologicalSequences_LargeChunk_SmallInput,
                   FSE_PathologicalSequences_LargeChunk_MediumInput,
                   FSE_PathologicalSequences_LargeChunk_LargeInput,
                   FSE_PathologicalSequences_LargeChunk_HugeInput,
                   FSE_PathologicalSequences_HugeChunk_TinyInput,
                   FSE_PathologicalSequences_HugeChunk_SmallInput,
                   FSE_PathologicalSequences_HugeChunk_MediumInput,
                   FSE_PathologicalSequences_HugeChunk_LargeInput,
                   FSE_PathologicalSequences_HugeChunk_HugeInput) {
  using T = TestType;

  constexpr int compression_level = 6;
  constexpr size_t warmup_iteration_count = 2;
  constexpr size_t total_iteration_count = 5;

  std::cout << "\n=====================================\n";
  std::cout << "Scenario : " << scenario_to_string(T::scenario) << "\n";
  std::cout << "Chunking : " << chunk_class_to_string(T::chunk_class) << "\n";
  std::cout << "Dataset  : " << data_size_class_to_string(T::data_size) << "\n";
  std::cout << "=====================================\n";

  const size_t chunk_size = chunk_size_from_class(T::chunk_class);

  auto data = generate_multi_test(T::scenario, T::data_size, chunk_size / 4);

  run_test(data, compression_level, warmup_iteration_count,
           total_iteration_count, chunk_size);
}

TEMPLATE_TEST_CASE(
    "decomp Zstd full mechanism x chunk x dataset matrix",
    "[zstd][nvcomp][matrix][mechanism]", Mixed_TextLike_TinyChunk_TinyInput,
    Mixed_TextLike_TinyChunk_SmallInput, Mixed_TextLike_TinyChunk_MediumInput,
    Mixed_TextLike_TinyChunk_LargeInput, Mixed_TextLike_TinyChunk_HugeInput,
    Mixed_TextLike_SmallChunk_TinyInput, Mixed_TextLike_SmallChunk_SmallInput,
    Mixed_TextLike_SmallChunk_MediumInput, Mixed_TextLike_SmallChunk_LargeInput,
    Mixed_TextLike_SmallChunk_HugeInput, Mixed_TextLike_MediumChunk_TinyInput,
    Mixed_TextLike_MediumChunk_SmallInput,
    Mixed_TextLike_MediumChunk_MediumInput,
    Mixed_TextLike_MediumChunk_LargeInput, Mixed_TextLike_MediumChunk_HugeInput,
    Mixed_TextLike_LargeChunk_TinyInput, Mixed_TextLike_LargeChunk_SmallInput,
    Mixed_TextLike_LargeChunk_MediumInput, Mixed_TextLike_LargeChunk_LargeInput,
    Mixed_TextLike_LargeChunk_HugeInput, Mixed_TextLike_HugeChunk_TinyInput,
    Mixed_TextLike_HugeChunk_SmallInput, Mixed_TextLike_HugeChunk_MediumInput,
    Mixed_TextLike_HugeChunk_LargeInput, Mixed_TextLike_HugeChunk_HugeInput,

    Mixed_BinaryLike_TinyChunk_TinyInput, Mixed_BinaryLike_TinyChunk_SmallInput,
    Mixed_BinaryLike_TinyChunk_MediumInput,
    Mixed_BinaryLike_TinyChunk_LargeInput, Mixed_BinaryLike_TinyChunk_HugeInput,
    Mixed_BinaryLike_SmallChunk_TinyInput,
    Mixed_BinaryLike_SmallChunk_SmallInput,
    Mixed_BinaryLike_SmallChunk_MediumInput,
    Mixed_BinaryLike_SmallChunk_LargeInput,
    Mixed_BinaryLike_SmallChunk_HugeInput,
    Mixed_BinaryLike_MediumChunk_TinyInput,
    Mixed_BinaryLike_MediumChunk_SmallInput,
    Mixed_BinaryLike_MediumChunk_MediumInput,
    Mixed_BinaryLike_MediumChunk_LargeInput,
    Mixed_BinaryLike_MediumChunk_HugeInput,
    Mixed_BinaryLike_LargeChunk_TinyInput,
    Mixed_BinaryLike_LargeChunk_SmallInput,
    Mixed_BinaryLike_LargeChunk_MediumInput,
    Mixed_BinaryLike_LargeChunk_LargeInput,
    Mixed_BinaryLike_LargeChunk_HugeInput, Mixed_BinaryLike_HugeChunk_TinyInput,
    Mixed_BinaryLike_HugeChunk_SmallInput,
    Mixed_BinaryLike_HugeChunk_MediumInput,
    Mixed_BinaryLike_HugeChunk_LargeInput, Mixed_BinaryLike_HugeChunk_HugeInput,

    Mixed_Bursty_TinyChunk_TinyInput, Mixed_Bursty_TinyChunk_SmallInput,
    Mixed_Bursty_TinyChunk_MediumInput, Mixed_Bursty_TinyChunk_LargeInput,
    Mixed_Bursty_TinyChunk_HugeInput, Mixed_Bursty_SmallChunk_TinyInput,
    Mixed_Bursty_SmallChunk_SmallInput, Mixed_Bursty_SmallChunk_MediumInput,
    Mixed_Bursty_SmallChunk_LargeInput, Mixed_Bursty_SmallChunk_HugeInput,
    Mixed_Bursty_MediumChunk_TinyInput, Mixed_Bursty_MediumChunk_SmallInput,
    Mixed_Bursty_MediumChunk_MediumInput, Mixed_Bursty_MediumChunk_LargeInput,
    Mixed_Bursty_MediumChunk_HugeInput, Mixed_Bursty_LargeChunk_TinyInput,
    Mixed_Bursty_LargeChunk_SmallInput, Mixed_Bursty_LargeChunk_MediumInput,
    Mixed_Bursty_LargeChunk_LargeInput, Mixed_Bursty_LargeChunk_HugeInput,
    Mixed_Bursty_HugeChunk_TinyInput, Mixed_Bursty_HugeChunk_SmallInput,
    Mixed_Bursty_HugeChunk_MediumInput, Mixed_Bursty_HugeChunk_LargeInput,
    Mixed_Bursty_HugeChunk_HugeInput) {
  using T = TestType;

  constexpr int compression_level = 6;
  constexpr size_t warmup_iteration_count = 2;
  constexpr size_t total_iteration_count = 5;

  std::cout << "\n=====================================\n";
  std::cout << "Scenario : " << scenario_to_string(T::scenario) << "\n";
  std::cout << "Chunking : " << chunk_class_to_string(T::chunk_class) << "\n";
  std::cout << "Dataset  : " << data_size_class_to_string(T::data_size) << "\n";
  std::cout << "=====================================\n";

  const size_t chunk_size = chunk_size_from_class(T::chunk_class);

  auto data = generate_multi_test(T::scenario, T::data_size, chunk_size / 4);

  run_test(data, compression_level, warmup_iteration_count,
           total_iteration_count, chunk_size);
}

TEMPLATE_TEST_CASE(
    "decomp Zstd full mechanism x chunk x dataset matrix",
    "[zstd][nvcomp][matrix][mechanism]", Random_Fully_TinyChunk_TinyInput,
    Random_Fully_TinyChunk_SmallInput, Random_Fully_TinyChunk_MediumInput,
    Random_Fully_TinyChunk_LargeInput, Random_Fully_TinyChunk_HugeInput,
    Random_Fully_SmallChunk_TinyInput, Random_Fully_SmallChunk_SmallInput,
    Random_Fully_SmallChunk_MediumInput, Random_Fully_SmallChunk_LargeInput,
    Random_Fully_SmallChunk_HugeInput, Random_Fully_MediumChunk_TinyInput,
    Random_Fully_MediumChunk_SmallInput, Random_Fully_MediumChunk_MediumInput,
    Random_Fully_MediumChunk_LargeInput, Random_Fully_MediumChunk_HugeInput,
    Random_Fully_LargeChunk_TinyInput, Random_Fully_LargeChunk_SmallInput,
    Random_Fully_LargeChunk_MediumInput, Random_Fully_LargeChunk_LargeInput,
    Random_Fully_LargeChunk_HugeInput, Random_Fully_HugeChunk_TinyInput,
    Random_Fully_HugeChunk_SmallInput, Random_Fully_HugeChunk_MediumInput,
    Random_Fully_HugeChunk_LargeInput, Random_Fully_HugeChunk_HugeInput,

    Random_Blockwise_TinyChunk_TinyInput, Random_Blockwise_TinyChunk_SmallInput,
    Random_Blockwise_TinyChunk_MediumInput,
    Random_Blockwise_TinyChunk_LargeInput, Random_Blockwise_TinyChunk_HugeInput,
    Random_Blockwise_SmallChunk_TinyInput,
    Random_Blockwise_SmallChunk_SmallInput,
    Random_Blockwise_SmallChunk_MediumInput,
    Random_Blockwise_SmallChunk_LargeInput,
    Random_Blockwise_SmallChunk_HugeInput,
    Random_Blockwise_MediumChunk_TinyInput,
    Random_Blockwise_MediumChunk_SmallInput,
    Random_Blockwise_MediumChunk_MediumInput,
    Random_Blockwise_MediumChunk_LargeInput,
    Random_Blockwise_MediumChunk_HugeInput,
    Random_Blockwise_LargeChunk_TinyInput,
    Random_Blockwise_LargeChunk_SmallInput,
    Random_Blockwise_LargeChunk_MediumInput,
    Random_Blockwise_LargeChunk_LargeInput,
    Random_Blockwise_LargeChunk_HugeInput, Random_Blockwise_HugeChunk_TinyInput,
    Random_Blockwise_HugeChunk_SmallInput,
    Random_Blockwise_HugeChunk_MediumInput,
    Random_Blockwise_HugeChunk_LargeInput,
    Random_Blockwise_HugeChunk_HugeInput) {
  using T = TestType;

  constexpr int compression_level = 6;
  constexpr size_t warmup_iteration_count = 2;
  constexpr size_t total_iteration_count = 5;

  std::cout << "\n=====================================\n";
  std::cout << "Scenario : " << scenario_to_string(T::scenario) << "\n";
  std::cout << "Chunking : " << chunk_class_to_string(T::chunk_class) << "\n";
  std::cout << "Dataset  : " << data_size_class_to_string(T::data_size) << "\n";
  std::cout << "=====================================\n";

  const size_t chunk_size = chunk_size_from_class(T::chunk_class);

  auto data = generate_multi_test(T::scenario, T::data_size, chunk_size / 4);

  run_test(data, compression_level, warmup_iteration_count,
           total_iteration_count, chunk_size);
}

TEMPLATE_TEST_CASE("decomp Zstd full mechanism x chunk x dataset matrix",
                   "[zstd][nvcomp][matrix][mechanism]",
                   Entropy_WorstCase_TinyChunk_TinyInput,
                   Entropy_WorstCase_TinyChunk_SmallInput,
                   Entropy_WorstCase_TinyChunk_MediumInput,
                   Entropy_WorstCase_TinyChunk_LargeInput,
                   Entropy_WorstCase_TinyChunk_HugeInput,
                   Entropy_WorstCase_SmallChunk_TinyInput,
                   Entropy_WorstCase_SmallChunk_SmallInput,
                   Entropy_WorstCase_SmallChunk_MediumInput,
                   Entropy_WorstCase_SmallChunk_LargeInput,
                   Entropy_WorstCase_SmallChunk_HugeInput,
                   Entropy_WorstCase_MediumChunk_TinyInput,
                   Entropy_WorstCase_MediumChunk_SmallInput,
                   Entropy_WorstCase_MediumChunk_MediumInput,
                   Entropy_WorstCase_MediumChunk_LargeInput,
                   Entropy_WorstCase_MediumChunk_HugeInput,
                   Entropy_WorstCase_LargeChunk_TinyInput,
                   Entropy_WorstCase_LargeChunk_SmallInput,
                   Entropy_WorstCase_LargeChunk_MediumInput,
                   Entropy_WorstCase_LargeChunk_LargeInput,
                   Entropy_WorstCase_LargeChunk_HugeInput,
                   Entropy_WorstCase_HugeChunk_TinyInput,
                   Entropy_WorstCase_HugeChunk_SmallInput,
                   Entropy_WorstCase_HugeChunk_MediumInput,
                   Entropy_WorstCase_HugeChunk_LargeInput,
                   Entropy_WorstCase_HugeChunk_HugeInput) {
  using T = TestType;

  constexpr int compression_level = 6;
  constexpr size_t warmup_iteration_count = 2;
  constexpr size_t total_iteration_count = 5;

  std::cout << "\n=====================================\n";
  std::cout << "Scenario : " << scenario_to_string(T::scenario) << "\n";
  std::cout << "Chunking : " << chunk_class_to_string(T::chunk_class) << "\n";
  std::cout << "Dataset  : " << data_size_class_to_string(T::data_size) << "\n";
  std::cout << "=====================================\n";

  const size_t chunk_size = chunk_size_from_class(T::chunk_class);

  auto data = generate_multi_test(T::scenario, T::data_size, chunk_size / 4);

  run_test(data, compression_level, warmup_iteration_count,
           total_iteration_count, chunk_size);
}
