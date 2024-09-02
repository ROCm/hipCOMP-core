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
#include "DeltaGPU.h"
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

} // namespace

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST_CASE("compress_10Thousand_Test", "[small]")
{
  size_t const n = 10000;

  using T = int32_t;

  T *input, *inputHost;
  size_t const numBytes = n * sizeof(*input);

  HIP_RT_CALL(hipMalloc((void**)&input, numBytes));

  HIP_RT_CALL(hipHostMalloc((void**)&inputHost, n * sizeof(*inputHost)));

  float const totalGB = numBytes / (1024.0 * 1024.0 * 1024.0);

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

  T *output, *outputHost;
  T** outputPtr;

  HIP_RT_CALL(hipMalloc((void**)&output, numBytes));
  HIP_RT_CALL(hipHostMalloc((void**)&outputHost, numBytes));

  HIP_RT_CALL(hipMalloc((void**)&outputPtr, sizeof(*outputPtr)));
  HIP_RT_CALL(hipMemcpy(
      outputPtr, &output, sizeof(*outputPtr), hipMemcpyHostToDevice));

  size_t* inputSizePtr;
  HIP_RT_CALL(hipMalloc((void**)&inputSizePtr, sizeof(*inputSizePtr)));
  HIP_RT_CALL(hipMemcpy(
      inputSizePtr, &n, sizeof(*inputSizePtr), hipMemcpyHostToDevice));

  void* workspace;
  size_t const workspaceSize = DeltaGPU::requiredWorkspaceSize(n, TypeOf<T>());
  HIP_RT_CALL(hipMalloc((void**)&workspace, workspaceSize));

  hipEvent_t start, stop;

  HIP_RT_CALL(hipEventCreate(&start));
  HIP_RT_CALL(hipEventCreate(&stop));
  HIP_RT_CALL(hipEventRecord(start, stream));

  DeltaGPU::compress(
      workspace,
      workspaceSize,
      TypeOf<T>(),
      (void**)outputPtr,
      input,
      inputSizePtr,
      2 * n,
      stream);
  HIP_RT_CALL(hipEventRecord(stop, stream));

  HIP_RT_CALL(hipStreamSynchronize(stream));
  float time;
  HIP_RT_CALL(hipEventElapsedTime(&time, start, stop));

  fromGPU(outputHost, output, n, stream);
  HIP_RT_CALL(hipStreamSynchronize(stream));
  HIP_RT_CALL(hipStreamDestroy(stream));

  HIP_RT_CALL(hipFree(output));
  HIP_RT_CALL(hipFree(outputPtr));
  HIP_RT_CALL(hipFree(inputSizePtr));

  // compute Delta on host
  std::vector<T> expected{inputHost[0]};

  for (size_t i = 1; i < n; ++i) {
    expected.emplace_back(inputHost[i] - inputHost[i - 1]);
  }

  // verify output
  for (size_t i = 0; i < n; ++i) {
    CHECK(expected[i] == outputHost[i]);
  }

  HIP_RT_CALL(hipHostFree(outputHost));

  HIP_RT_CALL(hipFree(input));
  HIP_RT_CALL(hipHostFree(inputHost));
}
