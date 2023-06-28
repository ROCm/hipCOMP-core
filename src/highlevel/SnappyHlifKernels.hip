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


#include "highlevel/SnappyHlifKernels.h"
#include "hipcomp_common_deps/hlif_shared.hiph"
#include "snappy/compression.hiph"
#include "snappy/decompression.hiph"
#include "HipUtils.h"

namespace hipcomp {

struct snappy_compress_wrapper : hlif_compress_wrapper {

private:
  hipcompStatus_t* status;

public:
  __device__ snappy_compress_wrapper(uint8_t* /*tmp_buffer*/, uint8_t* /*share_buffer*/, hipcompStatus_t* status)
   : status(status)
  {}
      
  __device__ void compress_chunk(
      uint8_t* tmp_output_buffer,
      const uint8_t* this_decomp_buffer,
      const size_t decomp_size,
      const size_t max_comp_chunk_size,
      size_t* comp_chunk_size) 
  {
    snappy::do_snap(
        this_decomp_buffer,
        decomp_size,
        tmp_output_buffer,
        max_comp_chunk_size,
        nullptr, // snappy status -- could add this later. Need to work through how to do error checking.
        comp_chunk_size);
  }

  __device__ hipcompStatus_t get_output_status() {
    return *status;
  }

  __device__ FormatType get_format_type() {
    return FormatType::Snappy;
  }
};

struct snappy_decompress_wrapper : hlif_decompress_wrapper {

private:
  hipcompStatus_t* status;

public:
  __device__ snappy_decompress_wrapper(uint8_t* /*shared_buffer*/, hipcompStatus_t* status)
    : status(status)
  {}
      
  __device__ void decompress_chunk(
      uint8_t* decomp_buffer,
      const uint8_t* comp_buffer,
      const size_t comp_chunk_size,
      const size_t decomp_buffer_size) 
  {
    snappy::do_unsnap(
        comp_buffer,
        comp_chunk_size,
        decomp_buffer,
        decomp_buffer_size,
        status,
        nullptr); // device_uncompressed_bytes -- unnecessary for HLIF
  }

  __device__ hipcompStatus_t get_output_status() {
    return *status;
  }
};

void snappyHlifBatchCompress(
    const CompressArgs& comp_args,
    const uint32_t max_ctas,
    hipStream_t stream) 
{
  const dim3 grid(max_ctas);
  const dim3 block(COMP_THREADS_PER_BLOCK);

  HlifCompressBatchKernel<snappy_compress_wrapper><<<grid, block, 0, stream>>>(
      comp_args);      
}

void snappyHlifBatchDecompress(
    const uint8_t* comp_buffer, 
    uint8_t* decomp_buffer, 
    const size_t raw_chunk_size,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t* comp_chunk_offsets,
    const size_t* comp_chunk_sizes,
    const uint32_t max_ctas,
    hipStream_t stream,
    hipcompStatus_t* output_status) 
{
  const dim3 grid(max_ctas);
  const dim3 block(DECOMP_THREADS_PER_BLOCK);
  HlifDecompressBatchKernel<snappy_decompress_wrapper><<<grid, block, 0, stream>>>(
      comp_buffer,
      decomp_buffer,
      raw_chunk_size,
      ix_chunk,
      num_chunks,
      comp_chunk_offsets,
      comp_chunk_sizes,
      output_status);
}

size_t snappyHlifCompMaxBlockOccupancy(const int device_id) 
{
  hipDeviceProp_t device_prop;
  hipGetDeviceProperties(&device_prop, device_id);
  int num_blocks_per_sm;
  constexpr int shmem_size = 0;
  hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm, 
      HlifCompressBatchKernel<snappy_compress_wrapper>, 
      COMP_THREADS_PER_BLOCK,
      shmem_size);
  
  return device_prop.multiProcessorCount * num_blocks_per_sm;
}

size_t snappyHlifDecompMaxBlockOccupancy(const int device_id) 
{
  hipDeviceProp_t device_prop;
  hipGetDeviceProperties(&device_prop, device_id);
  int num_blocks_per_sm;
  constexpr int shmem_size = 0;
  hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm, 
      HlifDecompressBatchKernel<snappy_decompress_wrapper, 1>, 
      DECOMP_THREADS_PER_BLOCK, 
      shmem_size);
  
  return device_prop.multiProcessorCount * num_blocks_per_sm;
}

} // hipcomp namespace
