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

#pragma once

#include "hip/hip_runtime.h"

#define nvcompAlignmentRequirements_t hipcompAlignmentRequirements_t
#define nvcompBatchedZstdDecompressAsync hipcompBatchedZstdDecompressAsync
#define nvcompBatchedZstdDecompressDefaultOpts                                 \
  hipcompBatchedZstdDecompressDefaultOpts
#define nvcompBatchedZstdDecompressGetRequiredAlignments                       \
  hipcompBatchedZstdDecompressGetRequiredAlignments
#define nvcompBatchedZstdDecompressGetTempSizeAsync                            \
  hipcompBatchedZstdDecompressGetTempSize
#define nvcompBatchedZstdDecompressOpts_t hipcompBatchedZstdDecompressOpts_t
#define nvcompStatus_t hipcompStatus_t

#define nvcompBatchedGzipDecompressAsync hipcompBatchedGzipDecompressAsync
#define nvcompBatchedGzipDecompressDefaultOpts                                 \
  hipcompBatchedGzipDecompressDefaultOpts
#define nvcompBatchedGzipDecompressGetRequiredAlignments                       \
  hipcompBatchedGzipDecompressGetRequiredAlignments
#define nvcompBatchedGzipDecompressGetTempSizeAsync                            \
  hipcompBatchedGzipDecompressGetTempSize
#define nvcompBatchedGzipDecompressOpts_t hipcompBatchedGzipDecompressOpts_t

#define nvcompBatchedDeflateDecompressAsync hipcompBatchedDeflateDecompressAsync
#define nvcompBatchedDeflateDecompressDefaultOpts                              \
  hipcompBatchedDeflateDecompressDefaultOpts
#define nvcompBatchedDeflateDecompressGetRequiredAlignments                    \
  hipcompBatchedDeflateDecompressGetRequiredAlignments
#define nvcompBatchedDeflateDecompressGetTempSizeAsync                         \
  hipcompBatchedDeflateDecompressGetTempSize
#define nvcompBatchedDeflateDecompressOpts_t                                   \
  hipcompBatchedDeflateDecompressOpts_t

#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaMalloc hipMalloc
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
