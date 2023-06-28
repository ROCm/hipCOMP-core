/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "tests/catch.hpp"
#include "RunLengthEncodeGPU.h"
#include "common.h"
#include "hipcomp.hpp"

#include "hip/hip_runtime.h"

#include <cstdlib>

#ifndef HIP_RT_CALL
#define HIP_RT_CALL(call)                                                     \
  {                                                                            \
    hipError_t hipStatus = call;                                             \
    if (hipSuccess != hipStatus) {                                           \
      fprintf(                                                                 \
          stderr,                                                              \
          "ERROR: HIP RT call \"%s\" in line %d of file %s failed with %s "   \
          "(%d).\n",                                                           \
          #call,                                                               \
          __LINE__,                                                            \
          __FILE__,                                                            \
          hipGetErrorString(hipStatus),                                      \
          hipStatus);                                                         \
      abort();                                                                 \
    }                                                                          \
  }
#endif

using namespace hipcomp;

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename T>
void toGPU(
    T* const output,
    T const* const input,
    size_t const num,
    hipStream_t stream)
{
  HIP_RT_CALL(hipMemcpyAsync(
      output, input, num * sizeof(T), hipMemcpyHostToDevice, stream));
}

template <typename T>
void fromGPU(
    T* const output,
    T const* const input,
    size_t const num,
    hipStream_t stream)
{
  HIP_RT_CALL(hipMemcpyAsync(
      output, input, num * sizeof(T), hipMemcpyDeviceToHost, stream));
}

template <typename T, typename V>
void compressAsyncTestRandom(const size_t n)
{
  T *input, *inputHost;
  size_t const numBytes = n * sizeof(*input);

  HIP_RT_CALL(hipMalloc((void**)&input, numBytes));

  HIP_RT_CALL(hipHostMalloc((void**)&inputHost, n * sizeof(*inputHost)));

  float const totalGB = numBytes / (1024.0f * 1024.0f * 1024.0f);

  hipStream_t stream;
  HIP_RT_CALL(hipStreamCreate(&stream));

  std::srand(0);

  T last = 0;
  for (size_t i = 0; i < n; ++i) {
    if (std::rand() % 3 == 0) {
      last = std::rand() % 1024;
    }
    inputHost[i] = last;
  }

  toGPU(input, inputHost, n, stream);

  T *outputValues, *outputValuesHost;
  V *outputCounts, *outputCountsHost;

  HIP_RT_CALL(hipMalloc((void**)&outputValues, sizeof(*outputValues) * n));
  HIP_RT_CALL(hipMalloc((void**)&outputCounts, sizeof(*outputCounts) * n));
  HIP_RT_CALL(
      hipHostMalloc((void**)&outputValuesHost, sizeof(*outputValuesHost) * n));
  HIP_RT_CALL(
      hipHostMalloc((void**)&outputCountsHost, sizeof(*outputCountsHost) * n));

  void* workspace;
  const size_t maxNum = 2 * n;
  size_t const workspaceSize = RunLengthEncodeGPU::requiredWorkspaceSize(
      maxNum, TypeOf<T>(), TypeOf<V>());
  HIP_RT_CALL(hipMalloc((void**)&workspace, workspaceSize));

  // create on device inputs
  size_t* numInDevice;
  size_t* numOutDevice;
  T** outputValuesPtr;
  V** outputCountsPtr;
  HIP_RT_CALL(hipMalloc((void**)&numInDevice, sizeof(*numInDevice)));
  HIP_RT_CALL(hipMalloc((void**)&numOutDevice, sizeof(*numOutDevice)));
  HIP_RT_CALL(hipMalloc((void**)&outputValuesPtr, sizeof(*outputValuesPtr)));
  HIP_RT_CALL(hipMalloc((void**)&outputCountsPtr, sizeof(*outputCountsPtr)));

  HIP_RT_CALL(hipMemcpy(
      numInDevice, &n, sizeof(*numInDevice), hipMemcpyHostToDevice));
  HIP_RT_CALL(hipMemcpy(
      outputValuesPtr,
      &outputValues,
      sizeof(outputValues),
      hipMemcpyHostToDevice));
  HIP_RT_CALL(hipMemcpy(
      outputCountsPtr,
      &outputCounts,
      sizeof(outputCounts),
      hipMemcpyHostToDevice));

  hipEvent_t start, stop;

  HIP_RT_CALL(hipEventCreate(&start));
  HIP_RT_CALL(hipEventCreate(&stop));
  HIP_RT_CALL(hipEventRecord(start, stream));

  RunLengthEncodeGPU::compressDownstream(
      workspace,
      workspaceSize,
      TypeOf<T>(),
      reinterpret_cast<void**>(outputValuesPtr),
      TypeOf<V>(),
      reinterpret_cast<void**>(outputCountsPtr),
      numOutDevice,
      input,
      numInDevice,
      maxNum,
      stream);
  HIP_RT_CALL(hipEventRecord(stop, stream));

  HIP_RT_CALL(hipStreamSynchronize(stream));
  float time;
  HIP_RT_CALL(hipEventElapsedTime(&time, start, stop));

  size_t numOut;
  HIP_RT_CALL(hipMemcpy(
      &numOut, numOutDevice, sizeof(numOut), hipMemcpyDeviceToHost));

  fromGPU(outputValuesHost, outputValues, numOut, stream);
  fromGPU(outputCountsHost, outputCounts, numOut, stream);
  HIP_RT_CALL(hipStreamSynchronize(stream));
  HIP_RT_CALL(hipStreamDestroy(stream));

  HIP_RT_CALL(hipFree(outputValues));
  HIP_RT_CALL(hipFree(outputCounts));
  HIP_RT_CALL(hipFree(outputValuesPtr));
  HIP_RT_CALL(hipFree(outputCountsPtr));
  HIP_RT_CALL(hipFree(numOutDevice));
  HIP_RT_CALL(hipFree(numInDevice));

  // compute RLE on host
  std::vector<T> expectedValues{inputHost[0]};
  std::vector<V> expectedCounts{1};

  for (size_t i = 1; i < n; ++i) {
    if (inputHost[i] == expectedValues.back()) {
      ++expectedCounts.back();
    } else {
      expectedValues.emplace_back(inputHost[i]);
      expectedCounts.emplace_back(1);
    }
  }

  REQUIRE(expectedCounts.size() == numOut);

  // verify output
  for (size_t i = 0; i < expectedCounts.size(); ++i) {
    if (!(expectedValues[i] == outputValuesHost[i]
          && expectedCounts[i] == outputCountsHost[i])) {
      std::cerr << "i = " << i << " exp " << (int64_t)expectedCounts[i] << ":"
                << (int64_t)expectedValues[i] << " act "
                << (int64_t)outputCountsHost[i] << ":"
                << (int64_t)outputValuesHost[i] << std::endl;
    }

    CHECK(expectedValues[i] == outputValuesHost[i]);
    CHECK(expectedCounts[i] == outputCountsHost[i]);
  }

  HIP_RT_CALL(hipHostFree(outputValuesHost));
  HIP_RT_CALL(hipHostFree(outputCountsHost));

  HIP_RT_CALL(hipFree(input));
  HIP_RT_CALL(hipHostFree(inputHost));
}

} // namespace

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST_CASE("compress_10Million_Test", "[small]")
{
  size_t const n = 10000000;

  using T = int32_t;
  using V = uint32_t;

  T *input, *inputHost;
  size_t const numBytes = n * sizeof(*input);

  HIP_RT_CALL(hipMalloc((void**)&input, numBytes));

  HIP_RT_CALL(hipHostMalloc((void**)&inputHost, n * sizeof(*inputHost)));

  float const totalGB = numBytes / (1024.0f * 1024.0f * 1024.0f);

  hipStream_t stream;
  HIP_RT_CALL(hipStreamCreate(&stream));

  std::srand(0);

  T last = 0;
  for (size_t i = 0; i < n; ++i) {
    if (std::rand() % 3 == 0) {
      last = std::rand() % 1024;
    }
    inputHost[i] = last;
  }

  toGPU(input, inputHost, n, stream);

  T *outputValues, *outputValuesHost;
  V *outputCounts, *outputCountsHost;

  HIP_RT_CALL(hipMalloc((void**)&outputValues, sizeof(*outputValues) * n));
  HIP_RT_CALL(hipMalloc((void**)&outputCounts, sizeof(*outputCounts) * n));
  HIP_RT_CALL(
      hipHostMalloc((void**)&outputValuesHost, sizeof(*outputValuesHost) * n));
  HIP_RT_CALL(
      hipHostMalloc((void**)&outputCountsHost, sizeof(*outputCountsHost) * n));

  size_t* numOutDevice;
  HIP_RT_CALL(hipMalloc((void**)&numOutDevice, sizeof(*numOutDevice)));

  void* workspace;
  size_t const workspaceSize
      = RunLengthEncodeGPU::requiredWorkspaceSize(n, TypeOf<T>(), TypeOf<V>());
  HIP_RT_CALL(hipMalloc((void**)&workspace, workspaceSize));

  hipEvent_t start, stop;

  HIP_RT_CALL(hipEventCreate(&start));
  HIP_RT_CALL(hipEventCreate(&stop));
  HIP_RT_CALL(hipEventRecord(start, stream));

  size_t numOut = 0;
  RunLengthEncodeGPU::compress(
      workspace,
      workspaceSize,
      TypeOf<T>(),
      outputValues,
      TypeOf<V>(),
      outputCounts,
      numOutDevice,
      input,
      n,
      stream);
  HIP_RT_CALL(hipEventRecord(stop, stream));

  HIP_RT_CALL(hipStreamSynchronize(stream));
  HIP_RT_CALL(hipMemcpy(
      &numOut, numOutDevice, sizeof(numOut), hipMemcpyDeviceToHost));

  float time;
  HIP_RT_CALL(hipEventElapsedTime(&time, start, stop));

  fromGPU(outputValuesHost, outputValues, numOut, stream);
  fromGPU(outputCountsHost, outputCounts, numOut, stream);
  HIP_RT_CALL(hipStreamSynchronize(stream));
  HIP_RT_CALL(hipStreamDestroy(stream));

  HIP_RT_CALL(hipFree(outputValues));
  HIP_RT_CALL(hipFree(outputCounts));

  // compute RLE on host
  std::vector<T> expectedValues{inputHost[0]};
  std::vector<V> expectedCounts{1};

  for (size_t i = 1; i < n; ++i) {
    if (inputHost[i] == expectedValues.back()) {
      ++expectedCounts.back();
    } else {
      expectedValues.emplace_back(inputHost[i]);
      expectedCounts.emplace_back(1);
    }
  }

  REQUIRE(expectedCounts.size() == numOut);

  // verify output
  for (size_t i = 0; i < expectedCounts.size(); ++i) {
    CHECK(expectedValues[i] == outputValuesHost[i]);
    CHECK(expectedCounts[i] == outputCountsHost[i]);
  }

  HIP_RT_CALL(hipHostFree(outputValuesHost));
  HIP_RT_CALL(hipHostFree(outputCountsHost));

  HIP_RT_CALL(hipFree(input));
  HIP_RT_CALL(hipHostFree(inputHost));
}

TEST_CASE("compressDownstream_10kUniform_Test", "[small]")
{
  using T = int32_t;
  using V = uint32_t;

  size_t const n = 10000;

  T *input, *inputHost;
  size_t const numBytes = n * sizeof(*input);

  HIP_RT_CALL(hipMalloc((void**)&input, numBytes));

  HIP_RT_CALL(hipHostMalloc((void**)&inputHost, n * sizeof(*inputHost)));

  float const totalGB = numBytes / (1024.0f * 1024.0f * 1024.0f);

  hipStream_t stream;
  HIP_RT_CALL(hipStreamCreate(&stream));

  T last = 37;
  for (size_t i = 0; i < n; ++i) {
    inputHost[i] = last;
  }

  toGPU(input, inputHost, n, stream);

  T *outputValues, *outputValuesHost;
  V *outputCounts, *outputCountsHost;

  HIP_RT_CALL(hipMalloc((void**)&outputValues, sizeof(*outputValues) * n));
  HIP_RT_CALL(hipMalloc((void**)&outputCounts, sizeof(*outputCounts) * n));
  HIP_RT_CALL(
      hipHostMalloc((void**)&outputValuesHost, sizeof(*outputValuesHost) * n));
  HIP_RT_CALL(
      hipHostMalloc((void**)&outputCountsHost, sizeof(*outputCountsHost) * n));

  void* workspace;
  const size_t maxNum = 2 * n;
  const size_t workspaceSize = RunLengthEncodeGPU::requiredWorkspaceSize(
      maxNum, TypeOf<T>(), TypeOf<V>());
  HIP_RT_CALL(hipMalloc((void**)&workspace, workspaceSize));

  // create on device inputs
  size_t* numInDevice;
  size_t* numOutDevice;
  T** outputValuesPtr;
  V** outputCountsPtr;
  HIP_RT_CALL(hipMalloc((void**)&numInDevice, sizeof(*numInDevice)));
  HIP_RT_CALL(hipMalloc((void**)&numOutDevice, sizeof(*numOutDevice)));
  HIP_RT_CALL(hipMalloc((void**)&outputValuesPtr, sizeof(*outputValuesPtr)));
  HIP_RT_CALL(hipMalloc((void**)&outputCountsPtr, sizeof(*outputCountsPtr)));

  HIP_RT_CALL(hipMemcpy(
      numInDevice, &n, sizeof(*numInDevice), hipMemcpyHostToDevice));
  HIP_RT_CALL(hipMemcpy(
      outputValuesPtr,
      &outputValues,
      sizeof(outputValues),
      hipMemcpyHostToDevice));
  HIP_RT_CALL(hipMemcpy(
      outputCountsPtr,
      &outputCounts,
      sizeof(outputCounts),
      hipMemcpyHostToDevice));

  hipEvent_t start, stop;

  HIP_RT_CALL(hipEventCreate(&start));
  HIP_RT_CALL(hipEventCreate(&stop));
  HIP_RT_CALL(hipEventRecord(start, stream));

  RunLengthEncodeGPU::compressDownstream(
      workspace,
      workspaceSize,
      TypeOf<T>(),
      reinterpret_cast<void**>(outputValuesPtr),
      TypeOf<V>(),
      reinterpret_cast<void**>(outputCountsPtr),
      numOutDevice,
      input,
      numInDevice,
      maxNum,
      stream);
  HIP_RT_CALL(hipEventRecord(stop, stream));

  HIP_RT_CALL(hipStreamSynchronize(stream));
  float time;
  HIP_RT_CALL(hipEventElapsedTime(&time, start, stop));

  size_t numOut;
  HIP_RT_CALL(hipMemcpy(
      &numOut, numOutDevice, sizeof(numOut), hipMemcpyDeviceToHost));

  fromGPU(outputValuesHost, outputValues, numOut, stream);
  fromGPU(outputCountsHost, outputCounts, numOut, stream);
  HIP_RT_CALL(hipStreamSynchronize(stream));
  HIP_RT_CALL(hipStreamDestroy(stream));

  HIP_RT_CALL(hipFree(outputValues));
  HIP_RT_CALL(hipFree(outputCounts));
  HIP_RT_CALL(hipFree(outputValuesPtr));
  HIP_RT_CALL(hipFree(outputCountsPtr));
  HIP_RT_CALL(hipFree(numOutDevice));
  HIP_RT_CALL(hipFree(numInDevice));

  // compute RLE on host
  std::vector<T> expectedValues{inputHost[0]};
  std::vector<V> expectedCounts{1};

  for (size_t i = 1; i < n; ++i) {
    if (inputHost[i] == expectedValues.back()) {
      ++expectedCounts.back();
    } else {
      expectedValues.emplace_back(inputHost[i]);
      expectedCounts.emplace_back(1);
    }
  }

  REQUIRE(expectedCounts.size() == numOut);

  // verify output
  for (size_t i = 0; i < expectedCounts.size(); ++i) {
    if (!(expectedValues[i] == outputValuesHost[i]
          && expectedCounts[i] == outputCountsHost[i])) {
      std::cerr << "i = " << i << " exp " << expectedCounts[i] << ":"
                << expectedValues[i] << " act " << outputCountsHost[i] << ":"
                << outputValuesHost[i] << std::endl;
    }

    CHECK(expectedValues[i] == outputValuesHost[i]);
    CHECK(expectedCounts[i] == outputCountsHost[i]);
  }

  HIP_RT_CALL(hipHostFree(outputValuesHost));
  HIP_RT_CALL(hipHostFree(outputCountsHost));

  HIP_RT_CALL(hipFree(input));
  HIP_RT_CALL(hipHostFree(inputHost));
}

TEST_CASE("compressDownstream_10k_16bit_count_Test", "[small]")
{
  const size_t n = 10003;

  compressAsyncTestRandom<uint8_t, uint16_t>(n);
  compressAsyncTestRandom<int8_t, uint16_t>(n);
  compressAsyncTestRandom<uint16_t, uint16_t>(n);
  compressAsyncTestRandom<int16_t, uint16_t>(n);
  compressAsyncTestRandom<int32_t, uint16_t>(n);
  compressAsyncTestRandom<uint32_t, uint16_t>(n);
  compressAsyncTestRandom<int64_t, uint16_t>(n);
  compressAsyncTestRandom<uint64_t, uint16_t>(n);
}

TEST_CASE("compressDownstream_10k_64bit_count_Test", "[small]")
{
  const size_t n = 10003;

  compressAsyncTestRandom<uint8_t, uint64_t>(n);
  compressAsyncTestRandom<int8_t, uint64_t>(n);
  compressAsyncTestRandom<uint16_t, uint64_t>(n);
  compressAsyncTestRandom<int16_t, uint64_t>(n);
  compressAsyncTestRandom<int32_t, uint64_t>(n);
  compressAsyncTestRandom<uint32_t, uint64_t>(n);
  compressAsyncTestRandom<int64_t, uint64_t>(n);
  compressAsyncTestRandom<uint64_t, uint64_t>(n);
}

TEST_CASE("compressDownstream_1024_32bit_count_Test", "[small]")
{
  for (size_t n = 512; n < 2048; ++n) {
    compressAsyncTestRandom<int32_t, uint16_t>(n);
  }
}
