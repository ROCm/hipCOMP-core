/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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

#define CATCH_CONFIG_MAIN

#include "hipcomp.hpp"
#include "hipcomp/bitcomp.h"

#include "catch.hpp"

#include <assert.h>
#include <stdlib.h>
#include <vector>

// Test GPU decompression with bitcomp batch API //

#ifdef ENABLE_BITCOMP

using namespace std;
using namespace hipcomp;

#define HIP_CHECK(cond)                                                       \
  do {                                                                         \
    hipError_t err = cond;                                                    \
    REQUIRE(err == hipSuccess);                                               \
  } while (false)

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

typedef enum
{
  EQUAL_LENGTH_BATCHES,
  INC_LENGTH_BATCHES,
  RANDOM_LENGTH_BATCHES
} split_type;

typedef enum
{
  RUNS,
  CST_Z,
  CST_FF,
  INC,
  RANDOM
} run_type;

typedef struct
{
  size_t offset;
  size_t size;
} batch_info;

// ****************************************************************************
// Generate a dataset of type T with size elements
template <typename T, run_type RUNTYPE>
std::vector<T> generate_data(size_t size)
{
    assert(size > 0);
    std::vector<T> input;
    if (RUNTYPE == RUNS)
    {
        T val = 0;
        size_t i = 0;
        while (true)
        {
            for (size_t ii = 0; ii < i; ii++)
            {
                input.push_back(val);
                if (input.size() == size)
                    return input;
            }
            val++;
            i++;
        }
    }
    if (RUNTYPE == CST_FF)
    {
        input.assign(size, static_cast<T>(-1));
        return input;
    }
    if (RUNTYPE == INC)
    {
        for (size_t i = 0; i < size; i++)
            input.push_back(static_cast<T>(i));
        return input;
    }
    if (RUNTYPE == RANDOM)
    {
        for (size_t i = 0; i < size; i++)
            input.push_back(static_cast<T>(rand()));
        return input;
    }
    // Default is zeroes
    input.assign(size, 0);
    return input;
}

// ****************************************************************************
// Generate offsets and sizes (in bytes) for each batch
template <typename T>
std::vector<batch_info> generate_offsets(split_type split, size_t size)
{
    std::vector<batch_info> v;
    batch_info info;
    if (split == EQUAL_LENGTH_BATCHES)
    {
        size_t length = sqrt(size);
        for (size_t i = 0; i < length; i++)
        {
            info.offset = i * length * sizeof(T);
            info.size = length * sizeof(T);
            v.push_back(info);
        }
        size_t remain = size - length * length;
        if (remain != 0)
        {
            info.offset = length * length * sizeof(T);
            info.size = remain * sizeof(T);
            v.push_back(info);
        }
    }
    if (split == INC_LENGTH_BATCHES)
    {
        size_t offset = 0;
        size_t inc = 1;
        while (offset < size)
        {
            info.offset = offset * sizeof(T);
            info.size = std::min(inc, size - offset) * sizeof(T);
            v.push_back(info);
            offset += inc++;
        }
    }
    if (split == RANDOM_LENGTH_BATCHES)
    {
        size_t length = sqrt(size);
        size_t offset = 0;
        while (offset < size)
        {
            int rnd = std::max(1, static_cast<int>(rand() % length));
            info.offset = offset * sizeof(T);
            info.size = std::min((size_t)rnd, size - offset) * sizeof(T);
            v.push_back(info);
            offset += rnd;
        }
    }
    return v;
}

/******************************************************************************
 * Main test ***********************************************************
 *****************************************************************************/

template <typename T>
void test_bitcomp_batch(
    std::vector<T> input, split_type split)
{
  int err = 0;
  size_t n = input.size();

  // Generate the batches according to the split
  std::vector<batch_info> offsets = generate_offsets<T>(split, n);

  size_t input_bytes = n * sizeof(T);
  size_t batches = offsets.size();

  // Copy the input data into a GPU buffer
  void *d_input_data;
  HIP_CHECK(hipMalloc(&d_input_data, input_bytes));
  HIP_CHECK(hipMemcpy(d_input_data, input.data(), input_bytes, hipMemcpyDefault));

  hipcompBatchedBitcompFormatOpts bitcomp_opts;
  bitcomp_opts.algorithm_type = 0; // Using default algorithm
  bitcomp_opts.data_type = TypeOf<T>();

  // Compute the output size of each batch and total size for the output buffer
  size_t total_output_size = 0;
  std::vector<size_t> batch_max_sizes(batches);
  for (size_t i = 0; i < batches; i++) {
    size_t maxi;
    hipcompBatchedBitcompCompressGetMaxOutputChunkSize(
        offsets[i].size, bitcomp_opts, &maxi);
    batch_max_sizes[i] = maxi;
    total_output_size += maxi;
  }
  void* d_comp_data;
  HIP_CHECK(hipMalloc(&d_comp_data, total_output_size));

  // Compute the input and output pointers for every batch
  std::vector<void*> input_ptrs(batches);
  std::vector<void*> comp_ptrs(batches);
  std::vector<size_t> input_sizes(batches);
  for (size_t i = 0; i < batches; i++) {
    input_ptrs[i] = reinterpret_cast<char*>(d_input_data) + offsets[i].offset;
    input_sizes[i] = offsets[i].size;
  }
  comp_ptrs[0] = reinterpret_cast<char*>(d_comp_data);
  for (size_t i = 1; i < batches; i++)
    comp_ptrs[i] = reinterpret_cast<char*>(comp_ptrs[i - 1]) + batch_max_sizes[i - 1];

  // Allocate device memory and copy the pointers and input sizes
  size_t pointer_bytes = batches * sizeof(void*);
  size_t batchsize_bytes = batches * sizeof(size_t);
  void **d_input_ptrs, **d_comp_ptrs;
  HIP_CHECK(hipMalloc(&d_input_ptrs, pointer_bytes));
  HIP_CHECK(hipMalloc(&d_comp_ptrs, pointer_bytes));
  size_t *d_input_sizes, *d_comp_sizes, *d_decomp_sizes;
  HIP_CHECK(hipMalloc((void**)&d_input_sizes, batchsize_bytes));
  HIP_CHECK(hipMalloc((void**)&d_comp_sizes, batchsize_bytes));
  HIP_CHECK(hipMalloc((void**)&d_decomp_sizes, batchsize_bytes));
  hipcompStatus_t* d_decomp_statuses;
  HIP_CHECK(hipMalloc((void**)&d_decomp_statuses, batches * sizeof(hipcompStatus_t)));
  HIP_CHECK(hipMemcpy(d_input_ptrs, input_ptrs.data(), pointer_bytes, hipMemcpyDefault));
  HIP_CHECK(hipMemcpy(d_comp_ptrs, comp_ptrs.data(), pointer_bytes, hipMemcpyDefault));
  HIP_CHECK(hipMemcpy(d_input_sizes, input_sizes.data(), batchsize_bytes, hipMemcpyDefault));
  std::vector<size_t> decomp_sizes(batches);

  hipStream_t stream;
  hipStreamCreate(&stream);

  // Compress async
  hipcompBatchedBitcompCompressAsync(
      d_input_ptrs,
      d_input_sizes,
      0,       // max_uncompressed_chunk_bytes: not used
      batches,
      nullptr, // device_temp_ptr: not used
      0,       // temp_bytes: not used
      d_comp_ptrs,
      d_comp_sizes,
      bitcomp_opts,
      stream);

  // Query the uncompressed sizes, make sure it matches the input sizes
  hipcompBatchedBitcompGetDecompressSizeAsync (d_comp_ptrs, d_comp_sizes, d_decomp_sizes, batches, stream);
  HIP_CHECK (hipStreamSynchronize (stream));
  HIP_CHECK(hipMemcpy(decomp_sizes.data(), d_decomp_sizes, batchsize_bytes, hipMemcpyDefault));
  REQUIRE (decomp_sizes == input_sizes);

  // Overwrite input and input sizes
  hipMemsetAsync(d_input_data, 0xee, input_bytes, stream);
  hipMemsetAsync(d_decomp_sizes, 0xee, batchsize_bytes, stream);

  // Decompress async, back into input
  hipcompBatchedBitcompDecompressAsync(
      d_comp_ptrs,
      nullptr, // device_compressed_bytes: not used
      d_input_sizes,
      d_decomp_sizes,
      batches,
      nullptr, // device_temp_ptr: not used
      0,       // temp_bytes: not used
      d_input_ptrs,
      d_decomp_statuses,
      stream);

  HIP_CHECK(hipStreamSynchronize(stream));

  // Copy the results back to CPU and check
  std::vector<T> res(input.size());
  HIP_CHECK(hipMemcpy(res.data(), d_input_data, input_bytes, hipMemcpyDefault));
  REQUIRE (res == input);
  HIP_CHECK(hipMemcpy(decomp_sizes.data(), d_decomp_sizes, batchsize_bytes, hipMemcpyDefault));
  REQUIRE (decomp_sizes == input_sizes);
  std::vector<hipcompStatus_t> decomp_statuses(batches);
  HIP_CHECK(hipMemcpy(decomp_statuses.data(), d_decomp_statuses,
                        batches * sizeof(hipcompStatus_t), hipMemcpyDefault));

  REQUIRE (decomp_statuses == std::vector<hipcompStatus_t>(batches, hipcompSuccess));

  HIP_CHECK(hipFree(d_input_data));
  HIP_CHECK(hipFree(d_comp_data));
  HIP_CHECK(hipFree(d_input_ptrs));
  HIP_CHECK(hipFree(d_comp_ptrs));
  HIP_CHECK(hipFree(d_input_sizes));
  HIP_CHECK(hipFree(d_comp_sizes));
  HIP_CHECK(hipFree(d_decomp_sizes));
  HIP_CHECK(hipFree(d_decomp_statuses));
}

// ****************************************************************************

template <typename T, run_type RUNTYPE>
void test_bitcomp_batch(size_t n)
{
  // Generate the data once for the given type and pattern
  std::vector<T> input = generate_data<T, RUNTYPE>(n);
  // Test different batch splits
  test_bitcomp_batch<T>(input, EQUAL_LENGTH_BATCHES);
  test_bitcomp_batch<T>(input, INC_LENGTH_BATCHES);
  test_bitcomp_batch<T>(input, RANDOM_LENGTH_BATCHES);
}

// ****************************************************************************

template <run_type RUNTYPE>
void test_bitcomp_batch (size_t n)
{
  // Test with different datatypes
  test_bitcomp_batch<uint8_t, RUNTYPE>(n);
  test_bitcomp_batch<int8_t, RUNTYPE>(n);
  test_bitcomp_batch<uint16_t, RUNTYPE>(n);
  test_bitcomp_batch<int16_t, RUNTYPE>(n);
  test_bitcomp_batch<uint32_t, RUNTYPE>(n);
  test_bitcomp_batch<int32_t, RUNTYPE>(n);
  test_bitcomp_batch<uint64_t, RUNTYPE>(n);
  test_bitcomp_batch<int64_t, RUNTYPE>(n);
}

} // namespace

/******************************************************************************
 * UNIT TESTS *****************************************************************
 *****************************************************************************/

#define SMALL 1000
#define LARGE 20000000

TEST_CASE("comp/decomp bitcomp-batch-runs small", "[hipcomp]")
{
    test_bitcomp_batch<RUNS> (SMALL);
}

TEST_CASE("comp/decomp bitcomp-batch-runs large", "[hipcomp]")
{
    test_bitcomp_batch<RUNS> (LARGE);
}

TEST_CASE("comp/decomp bitcomp-batch-zeroes small", "[hipcomp]")
{
    test_bitcomp_batch<CST_Z> (SMALL);
}

TEST_CASE("comp/decomp bitcomp-batch-zeroes large", "[hipcomp]")
{
    test_bitcomp_batch<CST_Z> (LARGE);
}

TEST_CASE("comp/decomp bitcomp-batch-ff small", "[hipcomp]")
{
    test_bitcomp_batch<CST_FF> (SMALL);
}

TEST_CASE("comp/decomp bitcomp-batch-ff large", "[hipcomp]")
{
    test_bitcomp_batch<CST_FF> (LARGE);
}

TEST_CASE("comp/decomp bitcomp-batch-inc small", "[hipcomp]")
{
    test_bitcomp_batch<INC> (SMALL);
}

TEST_CASE("comp/decomp bitcomp-batch-inc large", "[hipcomp]")
{
    test_bitcomp_batch<INC> (LARGE);
}

TEST_CASE("comp/decomp bitcomp-batch-random small", "[hipcomp]")
{
    test_bitcomp_batch<RANDOM> (SMALL);
}

TEST_CASE("comp/decomp bitcomp-batch-random large", "[hipcomp]")
{
    test_bitcomp_batch<RANDOM> (LARGE);
}

#endif // ENABLE_BITCOMP