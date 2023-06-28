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

#include "CascadedKernels.hiph"
#include "highlevel/CascadedHlifKernels.h"
#include "hipcomp_common_deps/hlif_shared.hiph"
#include "hipcomp/cascaded.h"
#include "HipUtils.h"

namespace hipcomp {

template <
    typename data_type,
    typename size_type,
    int threadblock_size,
    int chunk_size = default_chunk_size>
struct cascaded_compress_wrapper : hlif_compress_wrapper
{
private:
  hipcompStatus_t* status;
  const hipcompBatchedCascadedOpts_t options;

public:
  __device__ cascaded_compress_wrapper(
      const hipcompBatchedCascadedOpts_t options,
      uint8_t* /*tmp_buffer*/,
      uint8_t* /*share_buffer*/,
      hipcompStatus_t* status) :
      status(status), options(options)
  {}
      
  __device__ void compress_chunk(
      uint8_t* tmp_output_buffer,
      const uint8_t* this_decomp_buffer,
      const size_t decomp_size,
      const size_t max_comp_chunk_size,
      size_t* comp_chunk_size)
  {
    do_cascaded_compression_kernel<
        data_type,
        size_type,
        threadblock_size,
        chunk_size>(
        1,
        0,
        1,
        reinterpret_cast<const data_type* const*>(&this_decomp_buffer),
        &decomp_size,
        reinterpret_cast<void* const*>(&tmp_output_buffer),
        comp_chunk_size,
        options);
  }

  __device__ hipcompStatus_t& get_output_status() {
    return *status;
  }

  __device__ FormatType get_format_type() {
    return FormatType::Cascaded;
  }
};

template <
    typename data_type,
    typename size_type,
    int threadblock_size,
    int chunk_size = default_chunk_size>
struct cascaded_decompress_wrapper : hlif_decompress_wrapper
{

private:
  hipcompStatus_t* status;
  const hipcompBatchedCascadedOpts_t options;

public:
  __device__ cascaded_decompress_wrapper(
      const hipcompBatchedCascadedOpts_t options,
      uint8_t* /*shared_buffer*/,
      hipcompStatus_t* status) :
      status(status), options(options)
  {}
      
  __device__ void decompress_chunk(
      uint8_t* decomp_buffer,
      const uint8_t* comp_buffer,
      const size_t comp_chunk_size,
      const size_t decomp_buffer_size)
  {
    size_t actual_decompressed_bytes;
    hipcompStatus_t status;

    // allocate shmem and run fcn for data_type
    constexpr int shmem_size = compute_smem_size<
        chunk_size,
        sizeof(data_type),
        ((sizeof(data_type) <= 4) ? 4 : 8)>();
    __shared__ uint8_t shmem[shmem_size];

    cascaded_decompression_fcn<
        data_type,
        size_type,
        threadblock_size,
        chunk_size>(
        1,
        0,
        1,
        reinterpret_cast<const void* const*>(&comp_buffer),
        &comp_chunk_size,
        reinterpret_cast<void* const*>(&decomp_buffer),
        &decomp_buffer_size,
        &actual_decompressed_bytes,
        shmem,
        &status);
  }

  __device__ hipcompStatus_t& get_output_status() {
    return *status;
  }
};

void cascadedHlifBatchCompress(
    const CompressArgs& compress_args,
    const uint32_t max_ctas,
    hipStream_t stream,
    const hipcompBatchedCascadedOpts_t* options)
{
  const dim3 batch_size(max_ctas);
  constexpr int threadblock_size = cascaded_compress_threadblock_size;

  const hipcompType_t type = options->type;
  if (type == HIPCOMP_TYPE_CHAR || type == HIPCOMP_TYPE_UCHAR) {
    HlifCompressBatchKernel<
        cascaded_compress_wrapper<uint8_t, size_t, threadblock_size>,
        const hipcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(compress_args, *options);
  } else if (type == HIPCOMP_TYPE_SHORT || type == HIPCOMP_TYPE_USHORT) {
    HlifCompressBatchKernel<
        cascaded_compress_wrapper<uint16_t, size_t, threadblock_size>,
        const hipcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(compress_args, *options);
  } else if (type == HIPCOMP_TYPE_INT || type == HIPCOMP_TYPE_UINT) {
    HlifCompressBatchKernel<
        cascaded_compress_wrapper<uint32_t, size_t, threadblock_size>,
        const hipcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(compress_args, *options);
  } else if (type == HIPCOMP_TYPE_LONGLONG || type == HIPCOMP_TYPE_ULONGLONG) {
    HlifCompressBatchKernel<
        cascaded_compress_wrapper<uint64_t, size_t, threadblock_size>,
        const hipcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(compress_args, *options);
  }
}

void cascadedHlifBatchDecompress(
    const uint8_t* comp_buffer, 
    uint8_t* decomp_buffer, 
    const size_t raw_chunk_size,
    uint32_t* ix_chunk,
    const size_t num_chunks,
    const size_t* comp_chunk_offsets,
    const size_t* comp_chunk_sizes,
    const uint32_t max_ctas,
    hipStream_t stream,
    hipcompStatus_t* output_status,
    const hipcompBatchedCascadedOpts_t* options)
{
  const dim3 batch_size(max_ctas);
  constexpr int threadblock_size = cascaded_decompress_threadblock_size;

  const hipcompType_t type = options->type;
  if (type == HIPCOMP_TYPE_CHAR || type == HIPCOMP_TYPE_UCHAR) {
    HlifDecompressBatchKernel<
        cascaded_decompress_wrapper<uint8_t, size_t, threadblock_size>,
        1,
        const hipcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(
            comp_buffer,
            decomp_buffer,
            raw_chunk_size,
            ix_chunk,
            num_chunks,
            comp_chunk_offsets,
            comp_chunk_sizes,
            output_status,
            *options);
  } else if (type == HIPCOMP_TYPE_SHORT || type == HIPCOMP_TYPE_USHORT) {
    HlifDecompressBatchKernel<
        cascaded_decompress_wrapper<uint16_t, size_t, threadblock_size>,
        1,
        const hipcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(
            comp_buffer,
            decomp_buffer,
            raw_chunk_size,
            ix_chunk,
            num_chunks,
            comp_chunk_offsets,
            comp_chunk_sizes,
            output_status,
            *options);
  } else if (type == HIPCOMP_TYPE_INT || type == HIPCOMP_TYPE_UINT) {
    HlifDecompressBatchKernel<
        cascaded_decompress_wrapper<uint32_t, size_t, threadblock_size>,
        1,
        const hipcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(
            comp_buffer,
            decomp_buffer,
            raw_chunk_size,
            ix_chunk,
            num_chunks,
            comp_chunk_offsets,
            comp_chunk_sizes,
            output_status,
            *options);
  } else if (type == HIPCOMP_TYPE_LONGLONG || type == HIPCOMP_TYPE_ULONGLONG) {
    HlifDecompressBatchKernel<
        cascaded_decompress_wrapper<uint64_t, size_t, threadblock_size>,
        1,
        const hipcompBatchedCascadedOpts_t&>
        <<<batch_size, threadblock_size, 0, stream>>>(
            comp_buffer,
            decomp_buffer,
            raw_chunk_size,
            ix_chunk,
            num_chunks,
            comp_chunk_offsets,
            comp_chunk_sizes,
            output_status,
            *options);
  }

}

size_t cascadedHlifCompMaxBlockOccupancy(const int device_id, hipcompType_t type)
{
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, device_id);
  int numBlocksPerSM = 1;
  // This kernel only uses fixed-size shared memory, not shared memory
  // determined at kernel invocation time.
  constexpr int runtime_shmem_size = 0;
  constexpr int threadblock_size = cascaded_compress_threadblock_size;
  // The values will almost certainly be identical for all data types,
  // but just in case, handle types separately.
  if (type == HIPCOMP_TYPE_CHAR || type == HIPCOMP_TYPE_UCHAR) {
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifCompressBatchKernel<
            cascaded_compress_wrapper<uint8_t, size_t, threadblock_size>,
            const hipcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  } else if (type == HIPCOMP_TYPE_SHORT || type == HIPCOMP_TYPE_USHORT) {
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifCompressBatchKernel<
            cascaded_compress_wrapper<uint16_t, size_t, threadblock_size>,
            const hipcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  } else if (type == HIPCOMP_TYPE_INT || type == HIPCOMP_TYPE_UINT) {
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifCompressBatchKernel<
            cascaded_compress_wrapper<uint32_t, size_t, threadblock_size>,
            const hipcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  } else if (type == HIPCOMP_TYPE_LONGLONG || type == HIPCOMP_TYPE_ULONGLONG) {
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifCompressBatchKernel<
            cascaded_compress_wrapper<uint64_t, size_t, threadblock_size>,
            const hipcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  }

  return deviceProp.multiProcessorCount * numBlocksPerSM;
}

size_t cascadedHlifDecompMaxBlockOccupancy(
    const int device_id, hipcompType_t type)
{
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, device_id);
  int numBlocksPerSM = 1;
  // This kernel only uses fixed-size shared memory, not shared memory
  // determined at kernel invocation time.
  constexpr int runtime_shmem_size = 0;
  constexpr int threadblock_size = cascaded_decompress_threadblock_size;
  // The values will almost certainly be identical for all data types,
  // but just in case, handle types separately.
  if (type == HIPCOMP_TYPE_CHAR || type == HIPCOMP_TYPE_UCHAR) {
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifDecompressBatchKernel<
            cascaded_decompress_wrapper<uint8_t, size_t, threadblock_size>,
            1,
            const hipcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  } else if (type == HIPCOMP_TYPE_SHORT || type == HIPCOMP_TYPE_USHORT) {
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifDecompressBatchKernel<
            cascaded_decompress_wrapper<uint16_t, size_t, threadblock_size>,
            1,
            const hipcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  } else if (type == HIPCOMP_TYPE_INT || type == HIPCOMP_TYPE_UINT) {
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifDecompressBatchKernel<
            cascaded_decompress_wrapper<uint32_t, size_t, threadblock_size>,
            1,
            const hipcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  } else if (type == HIPCOMP_TYPE_LONGLONG || type == HIPCOMP_TYPE_ULONGLONG) {
    hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSM,
        HlifDecompressBatchKernel<
            cascaded_decompress_wrapper<uint64_t, size_t, threadblock_size>,
            1,
            const hipcompBatchedCascadedOpts_t&>,
        threadblock_size,
        runtime_shmem_size);
  }

  return deviceProp.multiProcessorCount * numBlocksPerSM;
}

} // hipcomp namespace
