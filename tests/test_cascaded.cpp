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
#include "hipcomp/cascaded.hpp"

#include "catch.hpp"

#include <assert.h>
#include <stdlib.h>
#include <vector>

// Test GPU decompression with cascaded compression API //

using namespace std;
using namespace hipcomp;

#define HIP_CHECK(cond)                                                       \
  do {                                                                         \
    hipError_t err = (cond);                                                  \
    REQUIRE(err == hipSuccess);                                               \
  } while (false)

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename T>
std::vector<T> buildRuns(const size_t numRuns, const size_t runSize)
{
  std::vector<T> input;
  for (size_t i = 0; i < numRuns; i++) {
    for (size_t j = 0; j < runSize; j++) {
      input.push_back(static_cast<T>(i));
    }
  }

  return input;
}

template <typename T>
void test_cascaded(const std::vector<T>& input, hipcompType_t data_type)
{
  // create GPU only input buffer
  T* d_in_data;
  const size_t in_bytes = sizeof(T) * input.size();
  HIP_CHECK(hipMalloc((void**)&d_in_data, in_bytes));
  HIP_CHECK(
      hipMemcpy(d_in_data, input.data(), in_bytes, hipMemcpyHostToDevice));

  hipStream_t stream;
  hipStreamCreate(&stream);

  hipcompBatchedCascadedOpts_t options = hipcompBatchedCascadedDefaultOpts;
  options.type = data_type;
  CascadedManager manager{options, stream};
  auto comp_config = manager.configure_compression(in_bytes);

  // Allocate output buffer
  uint8_t* d_comp_out;
  HIP_CHECK(hipMalloc(&d_comp_out, comp_config.max_compressed_buffer_size));

  manager.compress(
      reinterpret_cast<const uint8_t*>(d_in_data),
      d_comp_out,
      comp_config);

  HIP_CHECK(hipStreamSynchronize(stream));

  size_t comp_out_bytes = manager.get_compressed_output_size(d_comp_out);

  hipFree(d_in_data);

  // Test to make sure copying the compressed file is ok
  uint8_t* copied = 0;
  HIP_CHECK(hipMalloc(&copied, comp_out_bytes));
  HIP_CHECK(
      hipMemcpy(copied, d_comp_out, comp_out_bytes, hipMemcpyDeviceToDevice));
  hipFree(d_comp_out);
  d_comp_out = copied;

  auto decomp_config = manager.configure_decompression(d_comp_out);

  T* out_ptr;
  hipMalloc(&out_ptr, decomp_config.decomp_data_size);

  // make sure the data won't match input if not written to, so we can verify
  // correctness
  hipMemset(out_ptr, 0, decomp_config.decomp_data_size);

  manager.decompress(
      reinterpret_cast<uint8_t*>(out_ptr),
      d_comp_out,
      decomp_config);
  HIP_CHECK(hipStreamSynchronize(stream));

  // Copy result back to host
  std::vector<T> res(input.size());
  hipMemcpy(
      &res[0], out_ptr, input.size() * sizeof(T), hipMemcpyDeviceToHost);

  // Verify correctness
  REQUIRE(res == input);

  hipFree(d_comp_out);
  hipFree(out_ptr);
}

} // namespace

/******************************************************************************
 * UNIT TESTS *****************************************************************
 *****************************************************************************/

TEST_CASE("comp/decomp cascaded-small", "[hipcomp]")
{
  using T = int;

  std::vector<T> input = {0, 2, 2, 3, 0, 0, 0, 0, 0, 3, 1, 1, 1, 1, 1, 2, 3, 3};

  test_cascaded(input, HIPCOMP_TYPE_INT);
}

TEST_CASE("comp/decomp cascaded-1", "[hipcomp]")
{
  using T = int;

  const int num_elems = 500;
  std::vector<T> input;
  for (int i = 0; i < num_elems; ++i) {
    input.push_back(i >> 2);
  }

  test_cascaded(input, HIPCOMP_TYPE_INT);
}

TEST_CASE("comp/decomp cascaded-all-small-sizes", "[hipcomp][small]")
{
  using T = uint8_t;

  for (int total = 1; total < 4096; ++total) {
    std::vector<T> input = buildRuns<T>(total, 1);
    test_cascaded(input, HIPCOMP_TYPE_UCHAR);
  }
}

TEST_CASE("comp/decomp cascaded-multichunk", "[hipcomp][large]")
{
  using T = int;

  for (int total = 10; total < (1 << 24); total = total * 2 + 7) {
    std::vector<T> input = buildRuns<T>(total, 10);
    test_cascaded(input, HIPCOMP_TYPE_INT);
  }
}

TEST_CASE("comp/decomp cascaded-small-uint8", "[hipcomp][small]")
{
  using T = uint8_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_cascaded(input, HIPCOMP_TYPE_UCHAR);
  }
}

TEST_CASE("comp/decomp cascaded-small-uint16", "[hipcomp][small]")
{
  using T = uint16_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_cascaded(input, HIPCOMP_TYPE_USHORT);
  }
}

TEST_CASE("comp/decomp cascaded-small-uint32", "[hipcomp][small]")
{
  using T = uint32_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_cascaded(input, HIPCOMP_TYPE_UINT);
  }
}

TEST_CASE("comp/decomp cascaded-small-uint64", "[hipcomp][small]")
{
  using T = uint64_t;

  for (size_t num = 1; num < 1 << 18; num = num * 2 + 1) {
    std::vector<T> input = buildRuns<T>(num, 3);
    test_cascaded(input, HIPCOMP_TYPE_ULONGLONG);
  }
}

TEST_CASE("comp/decomp cascaded-none-aligned-sizes", "[hipcomp][small]")
{
  std::vector<size_t> input_sizes = { 1, 33, 1021 };

  std::vector<hipcompType_t> data_types = {
    HIPCOMP_TYPE_CHAR,
    HIPCOMP_TYPE_SHORT,
    HIPCOMP_TYPE_INT,
    HIPCOMP_TYPE_LONGLONG,
  };
  for (auto size : input_sizes) {
    std::vector<uint8_t> input = buildRuns<uint8_t>(1, size);
    for (auto type : data_types ) {
      test_cascaded(input, type);
    }
  }
}
