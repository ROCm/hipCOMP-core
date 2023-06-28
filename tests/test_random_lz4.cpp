/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
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

#ifndef VERBOSE
#define VERBOSE 0
#endif

#include "test_common.h"
#include "hipcomp/lz4.hpp"

// Test method that takes an input data, compresses it (on the CPU),
// decompresses it on the GPU, and verifies it is correct.
// Uses LZ4 Compression
template <typename T>
void test_lz4(const std::vector<T>& data, size_t /*chunk_size*/)
{
  const hipcompType_t type = hipcomp::TypeOf<T>();

  size_t chunk_size = 1 << 16;

#if VERBOSE > 1
  // dump input data
  std::cout << "Input" << std::endl;
  for (size_t i = 0; i < data.size(); i++)
    std::cout << data[i] << " ";
  std::cout << std::endl;
#endif

  // these two items will be the only forms of communication between
  // compression and decompression
  uint8_t* d_comp_out = nullptr;
  size_t comp_out_bytes = 0;

  hipStream_t stream;
  hipStreamCreate(&stream);

  {
    // this block handles compression, and we scope it to ensure only
    // serialized metadata and compressed data, are the only things passed
    // between compression and decopmression
    std::cout << "----------" << std::endl;
    std::cout << "uncompressed (B): " << data.size() * sizeof(T) << std::endl;

    // create GPU only input buffer
    uint8_t* d_in_data;
    const size_t in_bytes = sizeof(T) * data.size();
    HIP_CHECK(hipMalloc(&d_in_data, in_bytes));
    HIP_CHECK(
        hipMemcpy(d_in_data, data.data(), in_bytes, hipMemcpyHostToDevice));

    LZ4Manager lz4_manager(chunk_size, HIPCOMP_TYPE_CHAR, stream);
    
    auto comp_config = lz4_manager.configure_compression(in_bytes);
    HIP_CHECK(hipMalloc(&d_comp_out, comp_config.max_compressed_buffer_size));

    size_t* comp_out_bytes_ptr;
    HIP_CHECK(hipHostMalloc(
        (void**)&comp_out_bytes_ptr, sizeof(*comp_out_bytes_ptr)));

    lz4_manager.compress(
        d_in_data,
        d_comp_out,
        comp_config);

    HIP_CHECK(hipStreamSynchronize(stream));
    
    size_t comp_out_bytes = lz4_manager.get_compressed_output_size(d_comp_out);

    hipFree(d_in_data);

    std::cout << "comp_size: " << comp_out_bytes
              << ", compressed ratio: " << std::fixed << std::setprecision(2)
              << (double)in_bytes / comp_out_bytes << std::endl;
  }
  {
    // this block handles decompression, and we scope it to ensure only
    // the compressed data and the stream is passed
    // between compression and decompression

    LZ4Manager lz4_manager(chunk_size, HIPCOMP_TYPE_CHAR, stream);
    
    auto decomp_config = lz4_manager.configure_decompression(d_comp_out);

    const auto temp_bytes = lz4_manager.get_required_scratch_buffer_size();

    uint8_t* temp_ptr;
    hipMalloc(&temp_ptr, temp_bytes);
    lz4_manager.set_scratch_buffer(temp_ptr);

    uint8_t* out_ptr = NULL;
    hipMalloc(&out_ptr, decomp_config.decomp_data_size);

    lz4_manager.decompress(
        out_ptr,
        d_comp_out,
        decomp_config);

    HIP_CHECK(hipStreamSynchronize(stream));

    hipFree(d_comp_out);
    hipFree(temp_ptr);

    std::vector<T> res(decomp_config.decomp_data_size / sizeof(T));
    hipMemcpy(&res[0], out_ptr, decomp_config.decomp_data_size, hipMemcpyDeviceToHost);

#if VERBOSE > 1
    // dump output data
    std::cout << "Output" << std::endl;
    for (size_t i = 0; i < data.size(); i++)
      std::cout << ((T*)out_ptr)[i] << " ";
    std::cout << std::endl;
#endif

    REQUIRE(res == data);
  }
  hipStreamDestroy(stream);
}

template <typename T>
void test_random_lz4(
    int max_val,
    int max_run,
    size_t chunk_size)
{
  // generate random data
  std::vector<T> data;
  int seed = (max_val ^ max_run ^ static_cast<int>(chunk_size));
  random_runs(data, (T)max_val, (T)max_run, seed);

  test_lz4<T>(data, chunk_size);
}

// int
TEST_CASE("small-LZ4", "[small]")
{
  test_random_lz4<int>(10, 10, 10000);
}
TEST_CASE("medium-LZ4", "[small]")
{
  test_random_lz4<int>(10000, 10, 100000);
}

TEST_CASE("large-LZ4", "[large][bp]")
{
  test_random_lz4<int>(10000, 1000, 10000000);
}



// long long
TEST_CASE("small-LZ4-ll", "[small]")
{
  test_random_lz4<int64_t>(10, 10, 10000);
}
TEST_CASE("large-LZ4-ll", "[large]")
{
  test_random_lz4<int64_t>(10000, 1000, 10000000);
}
