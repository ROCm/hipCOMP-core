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

#include <hip/hip_runtime_api.h>
#include <string.h>

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

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

size_t compute_batch_size(const std::vector<std::vector<char>> &data,
                          const size_t chunk_size) {
  size_t batch_size = 0;
  for (size_t i = 0; i < data.size(); ++i) {
    const size_t num_chunks = (data[i].size() + chunk_size - 1) / chunk_size;
    batch_size += num_chunks;
  }

  return batch_size;
}

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

std::vector<std::vector<char>>
multi_file(const std::vector<std::string> &filenames) {
  std::vector<std::vector<char>> split_data;

  for (auto const &filename : filenames) {
    split_data.emplace_back(read_file(filename));
  }

  return split_data;
}

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
