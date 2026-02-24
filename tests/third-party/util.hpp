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
// Modifications Copyright (C) 2025 Advanced Micro Devices, Inc. All rights
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

#include <string.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef CUDA_BACKEND
#include "cuda_runtime_api.h"
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(func)                                                       \
  do {                                                                         \
    cudaError_t rt = (func);                                                   \
    if (rt != cudaSuccess) {                                                   \
      std::cout << "API call failure \"" #func "\" with " << rt << " at "      \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)
#endif // CUDA_CHECK

/**
 * Computes the total number of batches (chunks) across all input data vectors.
 * Each data vector is divided into chunks of the specified size.
 *
 * @param data Vector of input data vectors to be chunked
 * @param chunk_size Size of each chunk in bytes
 * @return Total number of chunks across all data vectors
 */
size_t compute_batch_size(const std::vector<std::vector<char>> &data,
                          const size_t chunk_size) {
  size_t batch_size = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    batch_size += num_chunks;
  }

  return batch_size;
}

/**
 * Computes the size of each chunk across all input data vectors.
 * Most chunks will be of the specified chunk_size, except for the last chunk
 * of each data vector which may be smaller if the data size is not evenly
 * divisible.
 *
 * @param data Vector of input data vectors
 * @param batch_size Total number of chunks (from compute_batch_size)
 * @param chunk_size Standard size for each chunk in bytes
 * @return Vector of sizes for each chunk
 */
std::vector<size_t>
compute_chunk_sizes(const std::vector<std::vector<char>> &data,
                    const size_t batch_size, const size_t chunk_size) {
  std::vector<size_t> sizes(batch_size, chunk_size);

  size_t offset = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    offset += num_chunks;
    if (data[i].size() % chunk_size != 0) {
      sizes[offset - 1] = data[i].size() % chunk_size;
    }
  }
  return sizes;
}

/**
 * Creates a vector of pointers to the beginning of each chunk across all data
 * vectors. Used for batched compression/decompression operations.
 *
 * @param data Vector of input data vectors
 * @param batch_size Total number of chunks
 * @param chunk_size Size of each chunk in bytes
 * @return Vector of void pointers pointing to the start of each chunk
 */
std::vector<void *> get_input_ptrs(const std::vector<std::vector<char>> &data,
                                   const size_t batch_size,
                                   const size_t chunk_size) {
  std::vector<void *> input_ptrs(batch_size);
  size_t chunk = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    for (size_t j = 0; j < num_chunks; ++j)
      input_ptrs[chunk++] = const_cast<void *>(
          static_cast<const void *>(data[i].data() + j * chunk_size));
  }
  return input_ptrs;
}

/**
 * Reads an entire file into a character vector.
 *
 * @param filename Path to the file to read
 * @return Vector containing the file contents
 * @throws std::runtime_error if the file cannot be opened
 */
std::vector<char> read_file(const std::string &filename) {
  std::ifstream fin(filename, std::ifstream::in | std::ifstream::binary |
                                  std::ifstream::ate);
  if (!fin) {
    throw std::runtime_error("Unable to open file: " + filename);
  }
  fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  // Query size
  size_t size = fin.tellg();
  fin.seekg(0, std::ifstream::beg);

  // Read the file
  std::vector<char> host_data(size);
  fin.read(host_data.data(), size);

  return host_data;
}

/**
 * Reads multiple files and returns their contents as separate vectors.
 *
 * @param filenames Vector of file paths to read
 * @return Vector of vectors, each containing the contents of one file
 */
std::vector<std::vector<char>>
multi_file(const std::vector<std::string> &filenames) {
  std::vector<std::vector<char>> split_data;

  for (auto const &filename : filenames) {
    split_data.emplace_back(read_file(filename));
  }

  return split_data;
}

/**
 * Rounds a number up to the nearest multiple of a given unit.
 *
 * @tparam U Type of the number to round
 * @tparam T Type of the unit
 * @param num The number to round up
 * @param unit The unit to round to
 * @return The smallest multiple of unit that is >= num
 */
template <typename U, typename T> U roundUpTo(const U num, const T unit) {
  return ((num + unit - 1) / unit) * unit;
}

/**
 * Produces is a literal of 62 characters that is repeated
 * until `length` is reached; last repetition may be cut off.
 *
 * \note Close to optimal compression ratios should
 *       be achievable with this input.
 */
std::vector<char> generate_text_sequence_repeat_62(size_t length) {
  const std::string charmap = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuv"
                              "wxyz0123456789"; // 62 characters

  std::vector<char> result;
  result.reserve(length);

  for (size_t i = 0; i < length; i++) {
    result.emplace_back(charmap[i % charmap.size()]);
  }
  return result;
}

/**
 * Generates multiple test sequences with varying lengths.
 * Each sequence i has length i * 100, all using the 62-character repeating
 * pattern.
 *
 * @param num_sequences Number of sequences to generate
 * @return Vector of test sequences with incrementing sizes
 */
std::vector<std::vector<char>>
generate_multi_test_sequence_repeat_62(const size_t num_sequences) {
  std::vector<std::vector<char>> split_data;
  split_data.reserve(num_sequences);
  for (size_t i = 1; i <= num_sequences; i++) {
    auto entry = generate_text_sequence_repeat_62(i * 100);
    split_data.emplace_back(entry);
  }
  return split_data;
}

enum class DataSizeClass {
  Tiny,   // fits in cache, kernel-launch dominated
  Small,  // latency-sensitive
  Medium, // steady-state throughput
  Large,  // amortized GPU execution
  Huge    // memory-bandwidth bound
};

/**
 * Converts a DataSizeClass enumeration to a human-readable string.
 *
 * @param c The DataSizeClass to convert
 * @return String description of the data size class
 */
const char *data_size_class_to_string(DataSizeClass c) {
  switch (c) {
  case DataSizeClass::Tiny:
    return "Tiny dataset (~64 KB)";
  case DataSizeClass::Small:
    return "Small dataset (~512 KB)";
  case DataSizeClass::Medium:
    return "Medium dataset (~4 MB)";
  case DataSizeClass::Large:
    return "Large dataset (~32 MB)";
  case DataSizeClass::Huge:
    return "Huge dataset (~256 MB)";
  }
  return "Unknown dataset size";
}

/**
 * Returns the total number of bytes for a given DataSizeClass.
 *
 * @param c The DataSizeClass to query
 * @return Number of bytes (64KB for Tiny, 512KB for Small, 4MB for Medium,
 *         32MB for Large, 256MB for Huge)
 */
constexpr size_t total_bytes_from_class(DataSizeClass c) {
  switch (c) {
  case DataSizeClass::Tiny:
    return 64 << 10; // 64 KB
  case DataSizeClass::Small:
    return 512 << 10; // 512 KB
  case DataSizeClass::Medium:
    return 4 << 20; // 4 MB
  case DataSizeClass::Large:
    return 32 << 20; // 32 MB
  case DataSizeClass::Huge:
    return 256 << 20; // 256 MB
  }
  return 4 << 20;
}

enum class ChunkClass {
  Tiny,   // fits in L1, kills LZ77 window reuse
  Small,  // typical message size
  Medium, // Zstd sweet spot
  Large,  // amortizes entropy tables
  Huge    // stresses GPU memory + windows
};

/**
 * Converts a ChunkClass enumeration to a human-readable string.
 *
 * @param c The ChunkClass to convert
 * @return String description of the chunk size class
 */
const char *chunk_class_to_string(ChunkClass c) {
  switch (c) {
  case ChunkClass::Tiny:
    return "Tiny chunks (1 KB)";
  case ChunkClass::Small:
    return "Small chunks (8 KB)";
  case ChunkClass::Medium:
    return "Medium chunks (64 KB)";
  case ChunkClass::Large:
    return "Large chunks (256 KB)";
  case ChunkClass::Huge:
    return "Huge chunks (1 MB)";
  }
  return "Unknown chunk size";
}

/**
 * Returns the chunk size in bytes for a given ChunkClass.
 *
 * @param c The ChunkClass to query
 * @return Chunk size (1KB for Tiny, 8KB for Small, 64KB for Medium,
 *         256KB for Large, 1MB for Huge)
 */
constexpr size_t chunk_size_from_class(ChunkClass c) {
  switch (c) {
  case ChunkClass::Tiny:
    return 1 << 10; // 1 KB
  case ChunkClass::Small:
    return 8 << 10; // 8 KB
  case ChunkClass::Medium:
    return 64 << 10; // 64 KB
  case ChunkClass::Large:
    return 256 << 10; // 256 KB
  case ChunkClass::Huge:
    return 1 << 20; // 1 MB
  }
  return 64 << 10;
}

enum class ZstdTestScenario {
  // LZ77 extremes
  LZ77_SingleByte_Long, // Tests LZ77 max match length (131072 bytes) and
                        // run-length encoding efficiency
  LZ77_ShortPattern,    // Tests LZ77 match finding overhead with frequent short
                        // matches and dictionary churn
  LZ77_LargeOffset,    // Tests LZ77 offset encoding costs and dictionary window
                       // access at large distances
  LZ77_WindowBoundary, // Tests LZ77 window sliding, boundary conditions, and
                       // cross-window match handling

  // Huffman-focused
  Huffman_Skewed_90_10, // Tests Huffman tree construction with skewed literal
                        // distribution (90% dominant symbol)
  Huffman_Skewed_99_1,  // Tests extreme Huffman code length optimization with
                        // 99% single-symbol dominance
  Huffman_Uniform_256, // Tests worst-case Huffman performance with flat entropy
                       // (all symbols equally likely)
  Huffman_Bimodal,     // Tests Huffman efficiency with two dominant symbols and
                       // minimal variety

  // FSE-focused
  FSE_ConstantSequences,      // Tests FSE table building for constant match
                              // lengths/offsets (best-case compression)
  FSE_SmallAlphabetSequences, // Tests FSE accuracy with limited symbol variety
                              // in sequence metadata
  FSE_VariableSequences,      // Tests FSE adaptivity to highly variable match
                              // lengths and offset distributions
  FSE_PathologicalSequences,  // Tests FSE worst-case: near-uniform sequence
                              // entropy requiring maximal table size

  // Mixed / realistic
  Mixed_TextLike, // Tests combined LZ77+Huffman+FSE pipeline on text-like data
                  // with repetition and variety
  Mixed_BinaryLike, // Tests handling of binary formats with null padding and
                    // structured/random sections
  Mixed_Bursty, // Tests zstd adaptivity: rapid transitions between compressible
                // and incompressible bursts

  // Baselines / worst cases
  Random_Fully,     // Baseline: fully random data to measure incompressible
                    // overhead and format costs
  Random_Blockwise, // Tests semi-structured random (constant blocks) to isolate
                    // block-level compression
  Entropy_WorstCase // Tests maximum entropy encoding overhead with
                    // deterministic pseudo-random patterns
};

/**
 * Converts a ZstdTestScenario enumeration to a human-readable string
 * description.
 *
 * @param s The ZstdTestScenario to convert
 * @return String description of the test scenario
 */
const char *scenario_to_string(ZstdTestScenario s) {
  switch (s) {
  case ZstdTestScenario::LZ77_SingleByte_Long:
    return "LZ77: single-byte long runs (max match length)";
  case ZstdTestScenario::LZ77_ShortPattern:
    return "LZ77: short repeating patterns (match churn)";
  case ZstdTestScenario::LZ77_LargeOffset:
    return "LZ77: large-distance matches (offset stress)";
  case ZstdTestScenario::LZ77_WindowBoundary:
    return "LZ77: window-boundary crossing matches";

  case ZstdTestScenario::Huffman_Skewed_90_10:
    return "Huffman: 90/10 skewed literals";
  case ZstdTestScenario::Huffman_Skewed_99_1:
    return "Huffman: 99/1 extreme skew";
  case ZstdTestScenario::Huffman_Uniform_256:
    return "Huffman: uniform 256-symbol distribution";
  case ZstdTestScenario::Huffman_Bimodal:
    return "Huffman: bimodal literal distribution";

  case ZstdTestScenario::FSE_ConstantSequences:
    return "FSE: constant match lengths and offsets";
  case ZstdTestScenario::FSE_SmallAlphabetSequences:
    return "FSE: small alphabet of sequence symbols";
  case ZstdTestScenario::FSE_VariableSequences:
    return "FSE: highly variable sequence metadata";
  case ZstdTestScenario::FSE_PathologicalSequences:
    return "FSE: near-uniform sequence entropy (worst)";

  case ZstdTestScenario::Mixed_TextLike:
    return "Mixed: text-like workload";
  case ZstdTestScenario::Mixed_BinaryLike:
    return "Mixed: binary-like workload";
  case ZstdTestScenario::Mixed_Bursty:
    return "Mixed: bursty compressible + noise";

  case ZstdTestScenario::Random_Fully:
    return "Baseline: fully random";
  case ZstdTestScenario::Random_Blockwise:
    return "Baseline: blockwise random";
  case ZstdTestScenario::Entropy_WorstCase:
    return "Worst-case entropy stress";
  }
  return "Unknown scenario";
}

/**
 * Template struct to combine scenario, chunk class, and data size into a single
 * tag. Used for compile-time test configuration.
 *
 * @tparam S The ZstdTestScenario
 * @tparam C The ChunkClass
 * @tparam D The DataSizeClass
 */
template <ZstdTestScenario S, ChunkClass C, DataSizeClass D>
struct FullMatrixTag {
  static constexpr ZstdTestScenario scenario = S;
  static constexpr ChunkClass chunk_class = C;
  static constexpr DataSizeClass data_size = D;
};

static std::mt19937 rng(12345);

/**
 * Generates a random byte value between 0 and 255.
 * Uses a Mersenne Twister random number generator with a fixed seed.
 *
 * @return A random byte value
 */
char random_byte() {
  static std::uniform_int_distribution<int> dist(0, 255);
  return static_cast<char>(dist(rng));
}

/**
 * Generates data consisting of a single repeated character.
 * Tests maximum match length in LZ77 compression.
 *
 * @param length Number of bytes to generate
 * @return Vector filled with the character 'A'
 */
std::vector<char> generate_lz77_single_char(size_t length) {
  return std::vector<char>(length, 'A');
}

/**
 * Generates data with repeating blocks of characters.
 * Each block cycles through the alphabet (A-Z).
 * Tests LZ77 match patterns at various offsets.
 *
 * @param length Total number of bytes to generate
 * @param block_size Size of the repeating block pattern (default: 64)
 * @return Vector with repeating block pattern
 */
std::vector<char> generate_lz77_block_repeat(size_t length,
                                             size_t block_size = 64) {
  std::vector<char> block(block_size);
  for (size_t i = 0; i < block_size; ++i)
    block[i] = 'A' + (i % 26);

  std::vector<char> out;
  out.reserve(length);
  while (out.size() < length)
    out.insert(out.end(), block.begin(), block.end());

  out.resize(length);
  return out;
}

/**
 * Generates fully random data with no patterns.
 * Represents incompressible data (worst case for compression).
 *
 * @param length Number of random bytes to generate
 * @return Vector of random bytes
 */
std::vector<char> generate_random(size_t length) {
  std::vector<char> out(length);
  for (auto &c : out)
    c = random_byte();
  return out;
}

/**
 * Generates data with a skewed symbol distribution.
 * 75% of bytes are 'A', 25% are random, which breaks LZ77 patterns
 * while creating a Huffman-favorable distribution.
 *
 * @param length Number of bytes to generate
 * @return Vector with skewed symbol distribution
 */
std::vector<char> generate_huffman_skewed(size_t length) {
  std::vector<char> out;
  out.reserve(length);

  std::uniform_int_distribution<int> rare(0, 255);

  for (size_t i = 0; i < length; ++i) {
    if (i % 4 == 0)
      out.push_back(static_cast<char>(rare(rng))); // break LZ77
    else
      out.push_back('A'); // dominant symbol
  }
  return out;
}

/**
 * Generates data with a uniform 256-symbol distribution.
 * Each byte position i has value (i & 0xFF), cycling through all possible byte
 * values. Tests Huffman encoding with flat entropy.
 *
 * @param length Number of bytes to generate
 * @return Vector with uniform symbol distribution
 */
std::vector<char> generate_huffman_uniform(size_t length) {
  std::vector<char> out(length);
  for (size_t i = 0; i < length; ++i)
    out[i] = static_cast<char>(i & 0xFF);
  return out;
}

/**
 * Generates data with an extreme skew: 99% dominant symbol, 1% random.
 * Every 100th byte is random, all others are 'A'.
 * Tests Huffman encoding with highly skewed symbol frequencies.
 *
 * @param length Number of bytes to generate
 * @return Vector with extreme symbol skew
 */
std::vector<char> generate_huffman_extreme(size_t length) {
  std::vector<char> out;
  out.reserve(length);

  std::uniform_int_distribution<int> rare(0, 255);

  for (size_t i = 0; i < length; ++i) {
    if ((i % 100) == 0) {
      out.push_back(static_cast<char>(rare(rng))); // rare symbol
    } else {
      out.push_back('A'); // dominant literal
    }
  }
  return out;
}

/**
 * Generates data with a bimodal symbol distribution.
 * Alternates between 'A' and 'B', with occasional random bytes to break LZ77.
 * Tests Huffman encoding with two dominant symbols.
 *
 * @param length Number of bytes to generate
 * @return Vector with bimodal distribution
 */
std::vector<char> generate_huffman_bimodal(size_t length) {
  std::vector<char> out;
  out.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    if (i & 1)
      out.push_back('A');
    else
      out.push_back('B');

    // break LZ77 every so often
    if ((i % 64) == 0)
      out.push_back(random_byte());
  }
  out.resize(length);
  return out;
}

/**
 * Generates data with constant sequence patterns.
 * Creates repeating blocks of identical characters to test
 * FSE encoding of match lengths and offsets.
 *
 * @param length Number of bytes to generate
 * @return Vector with constant sequence patterns
 */
std::vector<char> generate_fse_constant_sequences(size_t length) {
  const size_t block = 64;
  std::vector<char> out;
  out.reserve(length);

  std::vector<char> pattern(block, 'X');

  while (out.size() < length)
    out.insert(out.end(), pattern.begin(), pattern.end());

  out.resize(length);
  return out;
}

/**
 * Generates data with variable-length sequence patterns.
 * Random match lengths (8-135 bytes) of random characters, separated by noise.
 * Tests FSE encoding with highly variable sequence metadata.
 *
 * @param length Number of bytes to generate
 * @return Vector with variable sequence patterns
 */
std::vector<char> generate_fse_variable_sequences(size_t length) {
  std::vector<char> out;
  out.reserve(length);

  size_t pos = 0;
  while (pos < length) {
    size_t run = 8 + (rng() % 128); // random match length
    char c = static_cast<char>('A' + (rng() % 26));
    for (size_t i = 0; i < run && pos < length; ++i, ++pos)
      out.push_back(c);

    // break sequence
    if (pos < length)
      out.push_back(random_byte()), ++pos;
  }
  return out;
}

/**
 * Generates sequences with a small alphabet (A, B, C) and varying run lengths.
 * Tests FSE encoding with limited symbol variety but variable match patterns.
 *
 * @param length Number of bytes to generate
 * @return Vector with small alphabet sequences
 */
std::vector<char> generate_fse_small_alphabet(size_t length) {
  std::vector<char> out;
  out.reserve(length);

  const size_t run_lengths[] = {8, 16, 32};
  size_t pos = 0;

  while (pos < length) {
    size_t run = run_lengths[rng() % 3];
    char c = static_cast<char>('A' + (rng() % 3)); // small alphabet

    for (size_t i = 0; i < run && pos < length; ++i, ++pos)
      out.push_back(c);

    // break match slightly
    if (pos < length)
      out.push_back(random_byte()), ++pos;
  }
  return out;
}

/**
 * Generates binary-like data with null padding and random content.
 * First half of each 16-byte block is null (0x00), second half is random.
 * Simulates binary file formats with padding.
 *
 * @param length Number of bytes to generate
 * @return Vector with binary-like patterns
 */
std::vector<char> generate_binary_like(size_t length) {
  std::vector<char> out;
  out.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    if ((i % 16) < 8)
      out.push_back(0x00); // padding / nulls
    else
      out.push_back(random_byte());
  }
  return out;
}

/**
 * Generates bursty data alternating between compressible and incompressible
 * sections. Alternates 256-byte bursts of 'A' (compressible) with random data
 * (incompressible). Tests compression algorithm adaptivity.
 *
 * @param length Number of bytes to generate
 * @return Vector with bursty patterns
 */
std::vector<char> generate_bursty(size_t length) {
  std::vector<char> out;
  out.reserve(length);

  const size_t burst = 256;
  size_t pos = 0;

  while (pos < length) {
    // compressible burst
    for (size_t i = 0; i < burst && pos < length; ++i, ++pos)
      out.push_back('A');

    // incompressible burst
    for (size_t i = 0; i < burst && pos < length; ++i, ++pos)
      out.push_back(random_byte());
  }
  return out;
}

/**
 * Generates blockwise random data where each 64-byte block has a constant
 * value. The value changes between blocks but remains constant within each
 * block. Tests compression of semi-structured random data.
 *
 * @param length Number of bytes to generate
 * @return Vector with blockwise random pattern
 */
std::vector<char> generate_blockwise_random(size_t length) {
  std::vector<char> out;
  out.reserve(length);

  const size_t block = 64;
  char current = random_byte();

  for (size_t i = 0; i < length; ++i) {
    if ((i % block) == 0)
      current = random_byte();
    out.push_back(current);
  }
  return out;
}

/**
 * Generates realistic mixed data simulating text-like workloads.
 * 75% alphabet cycling (compressible), 25% random noise (incompressible).
 * Simulates typical real-world data with both LZ77 and Huffman opportunities.
 *
 * @param length Number of bytes to generate
 * @return Vector with mixed realistic patterns
 */
std::vector<char> generate_mixed_realistic(size_t length) {
  std::vector<char> out;
  out.reserve(length);

  for (size_t i = 0; i < length; ++i) {
    if (i % 128 < 96)
      out.push_back('A' + (i % 26)); // LZ + Huffman
    else
      out.push_back(random_byte()); // entropy noise
  }
  return out;
}

/**
 * Generates worst-case entropy data using a pseudo-random sequence.
 * Uses a multiplicative hash to create near-uniform entropy distribution
 * without true randomness, making it deterministic but incompressible.
 *
 * @param length Number of bytes to generate
 * @return Vector with worst-case entropy patterns
 */
std::vector<char> generate_entropy_worst_case(size_t length) {
  std::vector<char> out(length);
  for (size_t i = 0; i < length; ++i)
    out[i] = static_cast<char>((i * 1315423911u) & 0xFF);
  return out;
}

std::vector<char> generate_scenario(ZstdTestScenario s, size_t length) {
  switch (s) {

  // ---------- LZ77 ----------
  case ZstdTestScenario::LZ77_SingleByte_Long:
    return std::vector<char>(length, 'A');

  case ZstdTestScenario::LZ77_ShortPattern:
    return generate_lz77_block_repeat(length, 8);

  case ZstdTestScenario::LZ77_LargeOffset:
    return generate_lz77_block_repeat(length, 1024);

  case ZstdTestScenario::LZ77_WindowBoundary:
    return generate_lz77_block_repeat(length, 64 * 1024);

  // ---------- Huffman ----------
  case ZstdTestScenario::Huffman_Skewed_90_10:
    return generate_huffman_skewed(length);

  case ZstdTestScenario::Huffman_Skewed_99_1:
    return generate_huffman_extreme(length); // trivial variant

  case ZstdTestScenario::Huffman_Uniform_256:
    return generate_huffman_uniform(length);

  case ZstdTestScenario::Huffman_Bimodal:
    return generate_huffman_bimodal(length);

  // ---------- FSE ----------
  case ZstdTestScenario::FSE_ConstantSequences:
    return generate_fse_constant_sequences(length);

  case ZstdTestScenario::FSE_SmallAlphabetSequences:
    return generate_fse_small_alphabet(length);

  case ZstdTestScenario::FSE_VariableSequences:
    return generate_fse_variable_sequences(length);

  case ZstdTestScenario::FSE_PathologicalSequences:
    return generate_entropy_worst_case(length);

  // ---------- Mixed ----------
  case ZstdTestScenario::Mixed_TextLike:
    return generate_mixed_realistic(length);

  case ZstdTestScenario::Mixed_BinaryLike:
    return generate_binary_like(length);

  case ZstdTestScenario::Mixed_Bursty:
    return generate_bursty(length);

  // ---------- Baseline ----------
  case ZstdTestScenario::Random_Fully:
    return generate_random(length);

  case ZstdTestScenario::Random_Blockwise:
    return generate_blockwise_random(length);

  case ZstdTestScenario::Entropy_WorstCase:
    return generate_entropy_worst_case(length);
  }
  throw std::runtime_error("Unhandled scenario");
}

/**
 * Generates multiple test data chunks for a given scenario and data size class.
 * Creates progressively larger chunks (base_chunk_size * multiplier) until the
 * target total bytes for the data size class is reached.
 *
 * @param scenario The test scenario defining the data pattern
 * @param data_size_class The size class determining total data volume
 * @param base_chunk_size Base size for the first chunk (default: 256 bytes)
 * @return Vector of data chunks with increasing sizes
 */
std::vector<std::vector<char>>
generate_multi_test(ZstdTestScenario scenario, DataSizeClass data_size_class,
                    size_t base_chunk_size = 256) {
  const size_t target_total_bytes = total_bytes_from_class(data_size_class);

  std::vector<std::vector<char>> data;
  size_t accumulated = 0;
  size_t multiplier = 1;

  while (accumulated < target_total_bytes) {
    size_t this_size = base_chunk_size * multiplier;

    // Clamp last chunk so we don't overshoot too much
    if (accumulated + this_size > target_total_bytes) {
      this_size = target_total_bytes - accumulated;
    }

    data.emplace_back(generate_scenario(scenario, this_size));
    accumulated += this_size;
    ++multiplier;
  }

  return data;
}
