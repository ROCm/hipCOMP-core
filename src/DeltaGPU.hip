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

#include "DeltaGPU.h"
#include "common.h"
#include "type_macros.h"

#include <cassert>
#include <limits>

namespace hipcomp
{

/******************************************************************************
 * CONSTANTS ******************************************************************
 *****************************************************************************/

namespace
{

constexpr int const BLOCK_SIZE = 1024;

} // namespace

/******************************************************************************
 * KERNELS ********************************************************************
 *****************************************************************************/

namespace
{

template <typename VALUE>
__global__ void deltaKernel(
    VALUE** const outputPtr,
    const VALUE* const input,
    const size_t* const numDevice,
    const size_t /* maxNum */)
{
  const size_t num = *numDevice;

  if (BLOCK_SIZE * blockIdx.x < num) {
    VALUE* const output = *outputPtr;

    const int idx = threadIdx.x + BLOCK_SIZE * blockIdx.x;

    __shared__ VALUE buffer[BLOCK_SIZE + 1];

    if (idx < num) {
      buffer[threadIdx.x + 1] = input[idx];
    }

    if (threadIdx.x == 0) {
      // first thread must do something special
      if (idx > 0) {
        buffer[0] = input[idx - 1];
      } else {
        buffer[0] = 0;
      }
    }

    __syncthreads();

    if (idx < num) {
      output[idx] = buffer[threadIdx.x + 1] - buffer[threadIdx.x];
    }
  }
}

} // namespace

/******************************************************************************
 * HELPER FUNCTIONS ***********************************************************
 *****************************************************************************/

namespace
{

template <typename VALUE>
void deltaLaunch(
    void** const outPtr,
    void const* const in,
    const size_t* const numDevice,
    const size_t maxNum,
    hipStream_t stream)
{
  VALUE** const outTypedPtr = reinterpret_cast<VALUE**>(outPtr);
  const VALUE* const inTyped = static_cast<const VALUE*>(in);

  const dim3 block(BLOCK_SIZE);
  const dim3 grid(roundUpDiv(maxNum, BLOCK_SIZE));
  deltaKernel<<<grid, block, 0, stream>>>(
      outTypedPtr, inTyped, numDevice, maxNum);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error(
        "Failed to launch deltaKernel kernel: " + std::to_string(err));
  }
}

} // namespace

/******************************************************************************
 * PUBLIC STATIC METHODS ******************************************************
 *****************************************************************************/

void DeltaGPU::compress(
    void* const /* workspace */,
    const size_t /* workspaceSize*/,
    const hipcompType_t inType,
    void** const outPtr,
    const void* const in,
    const size_t* const numDevice,
    const size_t maxNum,
    hipStream_t stream)
{
  HIPCOMP_TYPE_ONE_SWITCH(
      inType, deltaLaunch, outPtr, in, numDevice, maxNum, stream);
}

size_t DeltaGPU::requiredWorkspaceSize(
    const size_t /*num*/, const hipcompType_t /* type */)
{
  return 0;
}

} // namespace hipcomp