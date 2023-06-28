// MIT License
// 
// Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef HIP_INCLUDE_HIP_HIP_RUNTIME_H
#define HIP_INCLUDE_HIP_HIP_RUNTIME_H

#if defined(__HIP_PLATFORM_NVCC__) || defined(__HIP_PLATFORM_NVIDIA__)
  #include "cuda_runtime.h"
  
  #define hipDevAttrComputeCapabilityMajor cudaDevAttrComputeCapabilityMajor
  #define hipDeviceGetAttribute cudaDeviceGetAttribute
  #define hipDeviceProp_t cudaDeviceProp
  #define hipDeviceSynchronize cudaDeviceSynchronize
  #define hipErrorInvalidValue cudaErrorInvalidValue
  #define hipError_t cudaError_t
  #define hipEventCreate cudaEventCreate
  #define hipEventElapsedTime cudaEventElapsedTime
  #define hipEventRecord cudaEventRecord
  #define hipEvent_t cudaEvent_t
  #define hipFree cudaFree
  #define hipFreeAsync cudaFreeAsync
  #define hipGetDeviceProperties cudaGetDeviceProperties
  #define hipGetErrorString cudaGetErrorString
  #define hipGetLastError cudaGetLastError
  #define hipHostFree cudaFreeHost
  #define hipHostMalloc cudaMallocHost
  #define hipHostMallocDefault cudaHostAllocDefault
  #define hipMalloc cudaMalloc
  #define hipMallocAsync cudaMallocAsync
  #define hipMallocManaged cudaMallocManaged
  #define hipMemcpy cudaMemcpy
  #define hipMemcpyAsync cudaMemcpyAsync
  #define hipMemcpyDefault cudaMemcpyDefault
  #define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
  #define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
  #define hipMemcpyHostToDevice cudaMemcpyHostToDevice
  #define hipMemcpyKind cudaMemcpyKind
  #define hipMemoryTypeDevice cudaMemoryTypeDevice
  #define hipMemset cudaMemset
  #define hipMemsetAsync cudaMemsetAsync
  #define hipOccupancyMaxActiveBlocksPerMultiprocessor cudaOccupancyMaxActiveBlocksPerMultiprocessor
  #define hipPointerAttribute_t cudaPointerAttributes
  #define hipPointerGetAttributes cudaPointerGetAttributes
  #define hipRuntimeGetVersion cudaRuntimeGetVersion
  #define hipStreamCreate cudaStreamCreate
  #define hipStreamDestroy cudaStreamDestroy
  #define hipStreamSynchronize cudaStreamSynchronize
  #define hipStream_t cudaStream_t
  #define hipSuccess cudaSuccess
#else
  #error only use for HIP/NVIDIA compile path!
#endif

#endif // HIP_INCLUDE_HIP_HIP_RUNTIME_H
