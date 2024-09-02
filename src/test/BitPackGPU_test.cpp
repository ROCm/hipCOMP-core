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
#include "BitPackGPU.h"
#include "common.h"
#include "hipcomp.hpp"
#include "unpack.h"

#include "hip/hip_runtime.h"

#include <cstdlib>
#include <limits>

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
void toGPU(T* const output, T const* const input, size_t const num)
{
  HIP_RT_CALL(
      hipMemcpy(output, input, num * sizeof(T), hipMemcpyHostToDevice));
}

template <typename T>
void fromGPU(T* const output, T const* const input, size_t const num)
{
  HIP_RT_CALL(
      hipMemcpy(output, input, num * sizeof(T), hipMemcpyDeviceToHost));
}

template <>
void
fromGPU<void>(void* const output, void const* const input, size_t const num)
{
  HIP_RT_CALL(hipMemcpy(output, input, num, hipMemcpyDeviceToHost));
}

template <typename T>
void runBitPackingOnGPU(
    T const* const inputHost,
    void* const outputHost,
    size_t const numBitsMax,
    size_t const n,
    size_t* const numBitsOut,
    T* const minValOut)
{
  T* input;

  HIP_RT_CALL(hipMalloc((void**)&input, n * sizeof(*input)));
  toGPU(input, inputHost, n);

  void* output;
  void** outputPtr;
  size_t const packedSize = (((numBitsMax * n) / 64U) + 1U) * 8U;

  size_t* numDevice;
  HIP_RT_CALL(hipMalloc((void**)&numDevice, sizeof(numDevice)));
  HIP_RT_CALL(
      hipMemcpy(numDevice, &n, sizeof(*numDevice), hipMemcpyHostToDevice));

  HIP_RT_CALL(hipMalloc(&output, packedSize));
  HIP_RT_CALL(hipMalloc(&outputPtr, sizeof(*outputPtr)));
  HIP_RT_CALL(
      hipMemcpy(outputPtr, &output, sizeof(output), hipMemcpyHostToDevice));
  HIP_RT_CALL(hipMemset(output, 0, packedSize));

  T* minValueDevice;
  HIP_RT_CALL(hipMalloc((void**)&minValueDevice, sizeof(*minValueDevice)));
  unsigned char* numBitsDevice;
  HIP_RT_CALL(hipMalloc((void**)&numBitsDevice, sizeof(*numBitsDevice)));

  T** minValueDevicePtr;
  HIP_RT_CALL(
      hipMalloc((void**)&minValueDevicePtr, sizeof(*minValueDevicePtr)));
  HIP_RT_CALL(hipMemcpy(
      minValueDevicePtr,
      &minValueDevice,
      sizeof(minValueDevice),
      hipMemcpyHostToDevice));
  unsigned char** numBitsDevicePtr;
  HIP_RT_CALL(
      hipMalloc((void**)&numBitsDevicePtr, sizeof(*numBitsDevicePtr)));
  HIP_RT_CALL(hipMemcpy(
      numBitsDevicePtr,
      &numBitsDevice,
      sizeof(numBitsDevice),
      hipMemcpyHostToDevice));

  void* workspace;
  size_t workspaceBytes = BitPackGPU::requiredWorkspaceSize(n, TypeOf<T>());
  HIP_RT_CALL(hipMalloc(&workspace, workspaceBytes));

  const hipcompType_t inType = TypeOf<T>();

  hipStream_t stream;
  HIP_RT_CALL(hipStreamCreate(&stream));

  BitPackGPU::compress(
      workspace,
      workspaceBytes,
      inType,
      outputPtr,
      input,
      numDevice,
      n,
      (void* const*)minValueDevicePtr,
      numBitsDevicePtr,
      stream);

  HIP_RT_CALL(hipStreamSynchronize(stream));
  HIP_RT_CALL(hipStreamDestroy(stream));

  fromGPU(minValOut, minValueDevice, 1);

  unsigned char numBits;
  fromGPU(&numBits, numBitsDevice, 1);
  *numBitsOut = numBits;

  fromGPU(outputHost, output, std::min(packedSize, n * sizeof(T)));

  HIP_RT_CALL(hipFree(input));
  HIP_RT_CALL(hipFree(output));
  HIP_RT_CALL(hipFree(outputPtr));
  HIP_RT_CALL(hipFree(workspace));
  HIP_RT_CALL(hipFree(minValueDevice));
  HIP_RT_CALL(hipFree(numBitsDevice));
  HIP_RT_CALL(hipFree(minValueDevicePtr));
  HIP_RT_CALL(hipFree(numBitsDevicePtr));
}

template<typename T>
void typeRangeTest()
{
  const size_t numBits = 8 * sizeof(T);
  size_t const n = 72351;
  std::vector<T> inputHost;
  inputHost.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    inputHost.emplace_back(static_cast<T>(i^static_cast<size_t>(0xfd956fda637535e7ULL)));
  }
  inputHost.front() = std::numeric_limits<T>::min();
  inputHost.back() = std::numeric_limits<T>::max();

  void* outputHost;

  size_t const numBytes = n * sizeof(*inputHost.data());
  HIP_RT_CALL(hipHostMalloc(&outputHost, numBytes));

  T minValue;
  size_t numBitsAct;
  runBitPackingOnGPU(inputHost.data(), outputHost, numBits, n, &numBitsAct, &minValue);

  REQUIRE(numBitsAct == numBits);

  // unpack
  std::vector<T> unpackedHost;
  for (size_t i = 0; i < n; ++i) {
    unpackedHost.emplace_back(
        unpackBytes(outputHost, static_cast<uint8_t>(numBitsAct), minValue, i));
  }

  // verify
  REQUIRE(unpackedHost.size() == n);
  for (size_t i = 0; i < n; ++i) {
    CHECK(unpackedHost[i] == inputHost[i]);
  }

  HIP_RT_CALL(hipHostFree(outputHost));
}

} // namespace

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST_CASE("compressInt16VarBitTest", "[small]")
{
  size_t const n = 10000;

  using T = int16_t;

  T const offset = 7231u;

  // generate a variety of random numbers
  std::vector<T> source(n);
  std::srand(0);
  //  for (T& v : source) {
  //    v = std::abs(static_cast<T>(std::rand())) %
  //    std::numeric_limits<T>::max();
  //  }

  T* inputHost;
  void* outputHost;

  size_t const numBytes = n * sizeof(*inputHost);
  HIP_RT_CALL(hipHostMalloc((void**)&inputHost, numBytes));
  HIP_RT_CALL(hipHostMalloc(&outputHost, numBytes));

  for (size_t numBits = 1; numBits < sizeof(T) * 8 - 1; ++numBits) {
    T minValue = 0;
    for (size_t i = 0; i < n; ++i) {
      inputHost[i] = (source[i] & ((1U << numBits) - 1)) + offset;
      if (i == 0 || inputHost[i] < minValue) {
        minValue = inputHost[i];
      }
    }

    T minValueAct;
    size_t numBitsAct;
    runBitPackingOnGPU(
        inputHost, outputHost, numBits, n, &numBitsAct, &minValueAct);

    REQUIRE(numBitsAct <= numBits);
    REQUIRE(minValueAct == minValue);

    // unpack
    std::vector<T> unpackedHost;
    for (size_t i = 0; i < n; ++i) {
      unpackedHost.emplace_back(unpackBytes(
          outputHost, static_cast<uint8_t>(numBitsAct), minValue, i));
    }

    // verify
    REQUIRE(unpackedHost.size() == n);
    for (size_t i = 0; i < n; ++i) {
      CHECK(unpackedHost[i] == inputHost[i]);
    }
  }

  HIP_RT_CALL(hipHostFree(outputHost));
  HIP_RT_CALL(hipHostFree(inputHost));
}

TEST_CASE("compressUint32VarBitTest", "[small]")
{
  size_t const n = 10000;
  int const offset = 87231;

  using T = uint32_t;

  // generate a variety of random numbers
  std::vector<T> source(n);
  std::srand(0);
  for (T& v : source) {
    v = static_cast<T>(std::rand()) % std::numeric_limits<T>::max();
  }

  T* inputHost;
  void* outputHost;

  size_t const numBytes = n * sizeof(*inputHost);
  HIP_RT_CALL(hipHostMalloc((void**)&inputHost, numBytes));
  HIP_RT_CALL(hipHostMalloc(&outputHost, numBytes));

  for (size_t numBits = 1; numBits < sizeof(T) * 8 - 1; ++numBits) {
    T minValue = 0;
    for (size_t i = 0; i < n; ++i) {
      inputHost[i] = (source[i] & ((1U << numBits) - 1)) + offset;
      if (i == 0 || inputHost[i] < minValue) {
        minValue = inputHost[i];
      }
    }

    T minValueAct;
    size_t numBitsAct;
    runBitPackingOnGPU(
        inputHost, outputHost, numBits, n, &numBitsAct, &minValueAct);

    REQUIRE(numBitsAct <= numBits);
    REQUIRE(minValueAct == minValue);

    // unpack
    std::vector<T> unpackedHost;
    for (size_t i = 0; i < n; ++i) {
      unpackedHost.emplace_back(unpackBytes(
          outputHost, static_cast<uint8_t>(numBitsAct), minValue, i));
    }

    // verify
    REQUIRE(unpackedHost.size() == n);
    for (size_t i = 0; i < n; ++i) {
      CHECK(unpackedHost[i] == inputHost[i]);
    }
  }

  HIP_RT_CALL(hipHostFree(outputHost));
  HIP_RT_CALL(hipHostFree(inputHost));
}

TEST_CASE("compressInt64VarBitTest", "[small]")
{
  size_t const n = 10000;
  int const offset = 87231;

  using T = int64_t;

  // generate a variety of random numbers
  std::vector<T> source(n);
  std::srand(0);
  for (T& v : source) {
    v = std::abs(static_cast<T>(std::rand())) % std::numeric_limits<T>::max();
  }

  T* inputHost;
  void* outputHost;

  size_t const numBytes = n * sizeof(*inputHost);
  HIP_RT_CALL(hipHostMalloc((void**)&inputHost, numBytes));
  HIP_RT_CALL(hipHostMalloc(&outputHost, numBytes));

  for (size_t numBits = 1; numBits < sizeof(T) * 8 - 1; ++numBits) {
    for (size_t i = 0; i < n; ++i) {
      inputHost[i] = (source[i] & ((1ULL << numBits) - 1)) + offset;
    }

    T minValue;
    size_t numBitsAct;
    runBitPackingOnGPU(
        inputHost, outputHost, numBits, n, &numBitsAct, &minValue);

    REQUIRE(numBitsAct <= numBits);

    // unpack
    std::vector<T> unpackedHost;
    for (size_t i = 0; i < n; ++i) {
      unpackedHost.emplace_back(unpackBytes(
          outputHost, static_cast<uint8_t>(numBitsAct), minValue, i));
    }

    // verify
    REQUIRE(unpackedHost.size() == n);
    for (size_t i = 0; i < n; ++i) {
      CHECK(unpackedHost[i] == inputHost[i]);
    }
  }

  HIP_RT_CALL(hipHostFree(outputHost));
  HIP_RT_CALL(hipHostFree(inputHost));
}

TEST_CASE("compressInt32VarSizeTest", "[large]")
{
  int const offset = 87231;
  size_t const numBits = 13;

  // unpack doesn't handle 0 bits
  std::vector<size_t> const sizes{2, 123, 3411, 83621, 872163, 100000001};

  using T = int32_t;

  // generate a variety of random numbers
  std::vector<T> source(sizes.back());
  std::srand(0);
  for (T& v : source) {
    v = std::abs(static_cast<T>(std::rand())) % std::numeric_limits<T>::max();
  }

  T* inputHost;
  void* outputHost;

  size_t const numBytes = sizes.back() * sizeof(*inputHost);
  HIP_RT_CALL(hipHostMalloc((void**)&inputHost, numBytes));
  HIP_RT_CALL(hipHostMalloc(&outputHost, numBytes));

  for (size_t const n : sizes) {
    for (size_t i = 0; i < n; ++i) {
      inputHost[i] = (source[i] & ((1U << numBits) - 1)) + offset;
    }

    T minValue;
    size_t numBitsAct;
    runBitPackingOnGPU(
        inputHost, outputHost, numBits, n, &numBitsAct, &minValue);

    REQUIRE(numBitsAct <= numBits);

    // unpack
    std::vector<T> unpackedHost;
    for (size_t i = 0; i < n; ++i) {
      unpackedHost.emplace_back(unpackBytes(
          outputHost, static_cast<uint8_t>(numBitsAct), minValue, i));
    }

    // verify
    REQUIRE(unpackedHost.size() == n);

    // checking 100 million entries can take a while, so sample instead
    size_t const numSamples = static_cast<size_t>(std::sqrt(n)) + 1;
    for (size_t i = 0; i < numSamples; ++i) {
      // only works for arrays less than 4 Billion.
      size_t const idx = static_cast<uint32_t>(source[i]) % n;
      CHECK(unpackedHost[idx] == inputHost[idx]);
    }
  }

  HIP_RT_CALL(hipHostFree(outputHost));
  HIP_RT_CALL(hipHostFree(inputHost));
}

TEST_CASE("compressInt64WideTest", "[small]")
{
  using T = int64_t;

  const size_t numBits = 40;

  // generate a variety of random numbers
  std::vector<T> source{
      100000511550L, 100000511550L, 100000511550L, 100000511550L,
      100000511550L, 100000511550L, 100000511550L, 100000511550L,
      100000511550L, 100000511550L, 999999704568L, 999999704568L,
      999999704568L, 999999704568L, 999999704568L, 999999704568L,
      999999704568L, 999999704568L, 999999704568L, 999999704568L};

  size_t const n = source.size();

  T* inputHost;
  void* outputHost;

  size_t const numBytes = n * sizeof(*inputHost);
  HIP_RT_CALL(hipHostMalloc((void**)&inputHost, numBytes));
  HIP_RT_CALL(hipHostMalloc(&outputHost, numBytes));

  memcpy(inputHost, source.data(), sizeof(*inputHost) * source.size());

  T minValue;
  size_t numBitsAct;
  runBitPackingOnGPU(inputHost, outputHost, numBits, n, &numBitsAct, &minValue);

  REQUIRE(numBitsAct == numBits);

  // unpack
  std::vector<T> unpackedHost;
  for (size_t i = 0; i < n; ++i) {
    unpackedHost.emplace_back(
        unpackBytes(outputHost, static_cast<uint8_t>(numBitsAct), minValue, i));
  }

  // verify
  REQUIRE(unpackedHost.size() == n);
  for (size_t i = 0; i < n; ++i) {
    CHECK(unpackedHost[i] == inputHost[i]);
  }

  HIP_RT_CALL(hipHostFree(outputHost));
  HIP_RT_CALL(hipHostFree(inputHost));
}

TEST_CASE("compressTypeInt8RangeTest", "[small]")
{
  typeRangeTest<int8_t>();
}

TEST_CASE("compressTypeInt16RangeTest", "[small]")
{
  typeRangeTest<int16_t>();
}

TEST_CASE("compressTypeInt32RangeTest", "[small]")
{
  typeRangeTest<int32_t>();
}

TEST_CASE("compressTypeInt64RangeTest", "[small]")
{
  typeRangeTest<int64_t>();
}

TEST_CASE("compressTypeUInt8RangeTest", "[small]")
{
  typeRangeTest<uint8_t>();
}

TEST_CASE("compressTypeUInt16RangeTest", "[small]")
{
  typeRangeTest<uint16_t>();
}

TEST_CASE("compressTypeUInt32RangeTest", "[small]")
{
  typeRangeTest<uint32_t>();
}

TEST_CASE("compressTypeUInt64RangeTest", "[small]")
{
  typeRangeTest<uint64_t>();
}
