// MIT License
//
// Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hip/hip_runtime_api.h"
#include "hipcomp/shared_types.h"

#include <vector>

/**
 * \brief Adjusts the entries of `bytes` to be a multiple of `padding_factor`.
 *
 * \param[in] bytes The number of uncompressed bytes (on the
 * device/host).
 *
 * \param[in] count Number of compressed blocks/chunks.
 *
 * \param[in] padding_factor
 * (optional) adds up to `padding_factor` additional bytes to the end of the
 * block if the current size is not a multiple of `padding_factor`. Defaults to
 * 4096 (4 KB). If you supply 0, no padding is applied. Note that the `bytes`
 * input is not updated by this method.
 *
 * \param[in] bytes_on_device If the 'bytes' input array is on the device (vs
 * host). Defaults to `false`.
 *
 * \note Currently downloads the `bytes` input to the host and reuploads if
 * `bytes` is an device array.
 */
hipcompStatus_t
hipcompPadDeviceBufferArraySizes(size_t *const bytes, // array of size_t
                                 int count, int padding_factor,
                                 bool bytes_on_device);

/**
 * \brief Allocate an device array of device buffers.
 *
 * Given are the number of bytes per buffer in an input array, which might
 * be located on the host or the device; see parameter `bytes_on_device`.
 *
 * \param[out] ptrs C-style return value. `ptrs` lives on the host, `*ptrs` and
 * `*ptrs[i]`, i in [0,count) hold device addresses.
 *
 * \param[in] bytes The number of uncompressed bytes (on the
 * device/host).
 *
 * \param[in] count Number of compressed blocks/chunks.
 *
 * \param[in] padding_factor
 * (optional) adds up to `padding_factor` additional bytes to the end of the
 * block if the current size is not a multiple of `padding_factor`. Defaults to
 * 4096 (4 KB). If you supply 0, no padding is applied. Note that the `bytes`
 * input is not updated by this method.
 *
 * \param[in] bytes_on_device If the 'bytes' input array is on the device (vs
 * host). Defaults to `false`.
 */
hipcompStatus_t
hipcompAllocateDeviceBufferArray(void ***ptrs, // address of array of void*
                                 size_t *const bytes, // array of size_t
                                 int count, int padding_factor = 4096,
                                 bool bytes_on_device = true);

/**
 * \brief Free a device array of device buffers.
 *
 * \note The caller is responsible to assign `ptrs` to `nullptr` after the call.
 *
 * \param[inout] ptrs One-dimensional array of device buffers. `ptrs` lives on
 * the host, the values of `ptrs` `ptrs[i]`, i in [0, count] are device
 * addresses.
 *
 * \param[in] count The number of device
 * buffers in the `ptrs` array.
 */
hipcompStatus_t hipcompFreeDeviceBufferArray(void **ptrs, // array of void*
                                             int count);

/**
 * \brief Reallocate decompression ouput buffers whose sizes have been
 * underestimated previously.
 *
 * \note The pointers to the reallocated device buffers will be updated in
 * `device_uncompressed_ptrs` as well as inserted into
 * `device_oom_uncompressed_ptrs`.
 *
 * \note The values in `device_uncompressed_bytes` will not be updated as the
 * correct number of bytes is already available in
 * device_actual_uncompressed_bytes. So the caller can use this variable instead
 * of the former where needed.
 *
 * \note Data in the replaced under-dimensioned buffers will not be copied into
 * the new larger buffer, it is lost.
 *
 * \param[in] device_uncompressed_ptrs The device buffers.
 *
 * \param[in] device_uncompressed_bytes
 * The estimated number of uncompressed bytes
 *
 * \param[in] device_actual_uncompressed_bytes Actual
 * number of uncompressed bytes, each buffer i with
 * 'device_uncompressed_bytes[i]' < 'actual_uncompressed_bytes[i]' where i =
 * 0,1,...,count-1 will be reallocated by this routine.
 *
 * \param[in] count Number of compressed blocks/chunks.
 *
 * \param[out] int oom_count Number of decompression buffers for which
 * out-of-memory (OOM) errors had been reported.
 *
 * \param[in] padding_factor
 * (optional) adds up to `padding_factor` additional bytes to the end of the
 * block if the current size is not a multiple of `padding_factor`. Defaults to
 * 4096 (4 KB). If you supply 0, no padding is applied. Note that the `bytes`
 * input is not updated by this method.
 *
 * \note If the reallocation becomes a bottleneck, one can modify the initial
 * estimate to reduce the chance for a reallocation. On the developer side, one
 * could think about moving some suboperations onto the device, e.g., the
 * operation that counts out-of-memory situations.
 */
hipcompStatus_t hipcompReallocateDecompressionBuffers(
    // outputs:
    int *oom_count,
    // inputs
    void **const device_uncompressed_ptrs,                // array of void*
    size_t *const device_uncompressed_bytes,              // array of size_t
    const size_t *const device_actual_uncompressed_bytes, // array of size_t
    int count,
    // optional inputs
    int padding_factor = 4096 // factor
);

/**
 * \brief Reallocate decompression ouput buffers whose sizes have been
 * underestimated previously (with additonal debug output).
 *
 * \note The pointers to the reallocated device buffers will be updated in
 * `device_uncompressed_ptrs` as well as inserted into
 * `device_oom_uncompressed_ptrs`.
 *
 * \note The values in `device_uncompressed_bytes` will not be updated as the
 * correct number of bytes is already available in
 * device_actual_uncompressed_bytes. So the caller can use this variable instead
 * of the former where needed.
 *
 * \note Data in the replaced under-dimensioned buffers will not be copied into
 * the new larger buffer, it is lost.
 *
 * \param[in] device_uncompressed_ptrs The device buffers.
 *
 * \param[in] device_uncompressed_bytes
 * The estimated number of uncompressed bytes
 *
 * \param[in] device_actual_uncompressed_bytes Actual
 * number of uncompressed bytes, each buffer i with
 * 'device_uncompressed_bytes[i]' < 'actual_uncompressed_bytes[i]' where i =
 * 0,1,...,count-1 will be reallocated by this routine.
 *
 * \param[in] count Number of compressed blocks/chunks.
 *
 * \param[out] int oom_count Number of decompression buffers for which
 * out-of-memory (OOM) errors had been reported.
 *
 * \param[out] device_oom_uncompressed_ptrs  decompression buffers for blocks
 * for which out-of-memory (OOM) errors had been reported. Pointer to array of
 * `void*`.
 *
 * \param[out] device_oom_uncompressed_bytes size of the reallocated
 * decompression buffers.
 *
 * \param[in] padding_factor
 * (optional) adds up to `padding_factor` additional bytes to the end of the
 * block if the current size is not a multiple of `padding_factor`. Defaults to
 * 4096 (4 KB). If you supply 0, no padding is applied. Note that the `bytes`
 * input is not updated by this method.
 *
 * \note If the reallocation becomes a bottleneck, one can modify the initial
 * estimate to reduce the chance for a reallocation. On the developer side, one
 * could think about moving some suboperations onto the device, e.g., the
 * operation that counts out-of-memory situations.
 */
hipcompStatus_t hipcompReallocateDecompressionBuffersDebug(
    // outputs:
    int *oom_count,
    void ***device_oom_uncompressed_ptrs,   // pointer to array of void*
    size_t **device_oom_uncompressed_bytes, // pointer to array of size_t
    // inputs
    void **const device_uncompressed_ptrs,                // array of void*
    size_t *const device_uncompressed_bytes,              // array of size_t
    const size_t *const device_actual_uncompressed_bytes, // array of size_t
    int count,
    // optional inputs
    int padding_factor = 4096 // factor
);
