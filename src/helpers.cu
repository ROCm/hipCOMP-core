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

/**
 * Helper routines for memory allocation etc.
 *
 * \note Has the `.hip` suffix as some operations might be performed partially
 * via GPU kernels in the future.
 */
#include "Check.h"
#include "hip/hip_runtime_api.h"
#include "hipcomp/helpers.h"

#include <vector>

namespace {
size_t padded_size(size_t size, size_t padding_factor) {
  if (padding_factor == 0)
    return size;
  return padding_factor *
         (size / padding_factor +
          (size % padding_factor >
           0)); // don't multiply out, relies on integer division
}
} // namespace

hipcompStatus_t
hipcompPadDeviceBufferArraySizes(size_t *const bytes, // array of size_t
                                 int count, int padding_factor,
                                 bool bytes_on_device) {
  // get num bytes per buffer
  std::vector<size_t> bytes_host(count);
  size_t bytes_bytes = count * sizeof(size_t);
  if (bytes_on_device) {
    CHECK_HIP_API_CALL(
        hipMemcpyDtoH((void *)bytes_host.data(), (void *)bytes, bytes_bytes));
  }
  size_t *bytes_ptr = (bytes_on_device) ? bytes_host.data() : bytes;
  for (int i = 0; i < count; i++) {
    size_t padded_bytes = ::padded_size(bytes_ptr[i], padding_factor);
    LOG_DEBUG(std::string("hipcompPadDeviceBufferArraySizes: pad buffer size "
                          "(size (B), unpadded size (B)): ") +
              std::to_string(padded_bytes) + ", " +
              std::to_string(bytes_ptr[i]));
    bytes_ptr[i] = padded_bytes;
  }
  if (bytes_on_device) {
    CHECK_HIP_API_CALL(
        hipMemcpyHtoD((void *)bytes, (void *)bytes_host.data(), bytes_bytes));
  }
  return hipcompSuccess;
}

hipcompStatus_t
hipcompAllocateDeviceBufferArray(void ***ptrs, // address of array of void*
                                 size_t *const bytes, // array of size_t
                                 int count, int padding_factor,
                                 bool bytes_on_device) {
  CHECK_NOT_NULL(ptrs);
  CHECK_NOT_NULL(bytes);

  // get num bytes per buffer
  size_t bytes_bytes = count * sizeof(size_t);
  std::vector<size_t> bytes_host(count);
  if (bytes_on_device) {
    CHECK_HIP_API_CALL(
        hipMemcpyDtoH((void *)bytes_host.data(), (void *)bytes, bytes_bytes));
  }
  size_t *bytes_ptr = (bytes_on_device) ? bytes_host.data() : bytes;

  // allocate buffers - device ptrs are first stored in host vector, then are
  // copied to device
  std::vector<void *> ptrs_host(count);
  bool padding_applied = false;
  for (int i = 0; i < count; i++) {
    size_t buffer_unpadded_bytes = bytes_ptr[i];
    size_t buffer_bytes = ::padded_size(buffer_unpadded_bytes, padding_factor);
    if (buffer_bytes != buffer_unpadded_bytes) {
      padding_applied = true;
    }
    LOG_DEBUG(std::string("hipcompAllocateDeviceBufferArray: allocate device "
                          "buffer (size (B), "
                          "unpadded size (B)): ") +
              std::to_string(buffer_bytes) + ", " +
              std::to_string(buffer_unpadded_bytes));
    bytes_ptr[i] = buffer_bytes;
    CHECK_HIP_API_CALL(hipMalloc(&ptrs_host[i], buffer_bytes));
  }
  const size_t ptrs_bytes = sizeof(void *) * count;
  CHECK_HIP_API_CALL(hipMalloc((void **)ptrs, ptrs_bytes));
  CHECK_HIP_API_CALL(
      hipMemcpyHtoD((void *)*ptrs, (void *)ptrs_host.data(), ptrs_bytes));
  if (bytes_on_device && padding_applied) {
    CHECK_HIP_API_CALL(
        hipMemcpyHtoD((void *)bytes, (void *)bytes_host.data(), bytes_bytes));
  }
  return hipcompSuccess;
}

hipcompStatus_t hipcompFreeDeviceBufferArray(void **ptrs, // array of void*
                                             int count) {
  CHECK_NOT_NULL(ptrs);

  std::vector<void *> ptrs_host(count);
  CHECK_HIP_API_CALL(hipMemcpyDtoH((void *)ptrs_host.data(), (void *)ptrs,
                                   count * sizeof(void *)));

  for (int i = 0; i < count; i++) {
    CHECK_NOT_NULL(ptrs_host[i]);
    CHECK_HIP_API_CALL(hipFree(ptrs_host[i]));
    ptrs[i] = nullptr;
  }

  CHECK_HIP_API_CALL(hipFree(ptrs));
  return hipcompSuccess;
}

hipcompStatus_t hipcompReallocateDecompressionBuffers(
    int *oom_count,
    // inout arguments:
    void **const device_uncompressed_ptrs,   // array of void*
    size_t *const device_uncompressed_bytes, // array of size_t
    // inputs
    const size_t *const device_actual_uncompressed_bytes, // array of size_t
    int count,
    // optional inputs:
    int padding_factor // factor
) {
  CHECK_NOT_NULL(device_uncompressed_ptrs);
  CHECK_NOT_NULL(device_uncompressed_bytes);
  CHECK_NOT_NULL(device_actual_uncompressed_bytes);

  // copy bytes arrays to the host
  std::vector<size_t> host_uncompressed_bytes(count);
  std::vector<size_t> host_actual_uncompressed_bytes(count);

  CHECK_HIP_API_CALL(hipMemcpyDtoH((void *)host_uncompressed_bytes.data(),
                                   (void *)device_uncompressed_bytes,
                                   sizeof(size_t) * count));
  CHECK_HIP_API_CALL(hipMemcpyDtoH(
      (void *)host_actual_uncompressed_bytes.data(),
      (void *)device_actual_uncompressed_bytes, sizeof(size_t) * count));

  // count out-of-memory instances
  *oom_count = 0;
  for (int i = 0; i < count; i++) {
    if (host_uncompressed_bytes[i] <
        host_actual_uncompressed_bytes[i]) { // indicates insufficient memory
      (*oom_count)++;
    }
  }

  LOG_DEBUG(std::string("hipcompReallocateDecompressionBuffers: number of "
                        "insufficiently sized buffers: ") +
            std::to_string(*oom_count));

  if (oom_count == 0)
    return hipcompSuccess;

  // allocate, download host data structures
  std::vector<void *> host_uncompressed_ptrs(count);
  CHECK_HIP_API_CALL(hipMemcpyDtoH(host_uncompressed_ptrs.data(),
                                   device_uncompressed_ptrs,
                                   sizeof(void *) * count));

  // free old buffers, allocate new buffers
  int n = 0;
  for (int i = 0; i < count; i++) {
    if (host_uncompressed_bytes[i] <
        host_actual_uncompressed_bytes[i]) { // indicates insufficient memory
      size_t old_bytes = host_uncompressed_bytes[i];
      size_t new_bytes =
          ::padded_size(host_actual_uncompressed_bytes[i], padding_factor);

      // allocate a new buffer
      void *new_ptr = nullptr;
      CHECK_HIP_API_CALL(hipMalloc((void **)&new_ptr, new_bytes));

      // reallocate pointer
      CHECK_HIP_API_CALL(hipFree(host_uncompressed_ptrs[i]));
      host_uncompressed_ptrs[i] = new_ptr;
      host_uncompressed_bytes[i] = new_bytes;

      LOG_DEBUG(std::string("hipcompReallocateDecompressionBuffers: "
                            "reallocated buffer (#, old "
                            "#bytes, new #bytes): ") +
                std::to_string(i) + ", " + std::to_string(old_bytes) + ", " +
                std::to_string(new_bytes));
      n++;
    }
  }

  // copy host buffer arrays to device
  const auto ptrs_bytes = sizeof(void *) * count;
  size_t bytes_bytes = count * sizeof(size_t);

  CHECK_HIP_API_CALL(hipMemcpyHtoD((void *)device_uncompressed_ptrs,
                                   (void *)host_uncompressed_ptrs.data(),
                                   ptrs_bytes));
  CHECK_HIP_API_CALL(hipMemcpyHtoD((void *)device_uncompressed_bytes,
                                   (void *)host_uncompressed_bytes.data(),
                                   bytes_bytes));
  return hipcompSuccess;
}

hipcompStatus_t hipcompReallocateDecompressionBuffersDebug(
    // outputs:
    int *oom_count,
    void ***device_oom_uncompressed_ptrs,   // pointer to array of void*
    size_t **device_oom_uncompressed_bytes, // pointer to array of size_t
    // inout arguments:
    void **const device_uncompressed_ptrs,   // array of void*
    size_t *const device_uncompressed_bytes, // array of size_t
    // inputs
    const size_t *const device_actual_uncompressed_bytes, // array of size_t
    int count,
    // optional inputs:
    int padding_factor // factor
) {
  CHECK_NOT_NULL(device_uncompressed_ptrs);
  CHECK_NOT_NULL(device_uncompressed_bytes);
  CHECK_NOT_NULL(device_actual_uncompressed_bytes);
  CHECK_NOT_NULL(oom_count);
  CHECK_NOT_NULL(device_oom_uncompressed_ptrs);
  CHECK_NOT_NULL(device_oom_uncompressed_bytes);

  // copy bytes arrays to the host
  std::vector<size_t> host_uncompressed_bytes(count);
  std::vector<size_t> host_actual_uncompressed_bytes(count);

  CHECK_HIP_API_CALL(hipMemcpyDtoH((void *)host_uncompressed_bytes.data(),
                                   (void *)device_uncompressed_bytes,
                                   sizeof(size_t) * count));
  CHECK_HIP_API_CALL(hipMemcpyDtoH(
      (void *)host_actual_uncompressed_bytes.data(),
      (void *)device_actual_uncompressed_bytes, sizeof(size_t) * count));

  // count out-of-memory instances
  *oom_count = 0;
  for (int i = 0; i < count; i++) {
    if (host_uncompressed_bytes[i] <
        host_actual_uncompressed_bytes[i]) { // indicates insufficient memory
      (*oom_count)++;
    }
  }

  LOG_DEBUG(std::string("hipcompReallocateDecompressionBuffers: number of "
                        "insufficiently sized buffers: ") +
            std::to_string(*oom_count));

  if (oom_count == 0)
    return hipcompSuccess;

  // allocate, download host data structures
  std::vector<void *> host_oom_uncompressed_ptrs(*oom_count);
  std::vector<size_t> host_oom_uncompressed_bytes(*oom_count);

  std::vector<void *> host_uncompressed_ptrs(count);
  CHECK_HIP_API_CALL(hipMemcpyDtoH(host_uncompressed_ptrs.data(),
                                   device_uncompressed_ptrs,
                                   sizeof(void *) * count));

  // free old buffers, allocate new buffers
  int n = 0;
  for (int i = 0; i < count; i++) {
    if (host_uncompressed_bytes[i] <
        host_actual_uncompressed_bytes[i]) { // indicates insufficient memory
      size_t old_bytes = host_uncompressed_bytes[i];
      size_t new_bytes =
          ::padded_size(host_actual_uncompressed_bytes[i], padding_factor);

      // allocate a new buffer
      void *new_ptr = nullptr;
      CHECK_HIP_API_CALL(hipMalloc((void **)&new_ptr, new_bytes));

      // reallocate pointer
      CHECK_HIP_API_CALL(hipFree(host_uncompressed_ptrs[i]));
      host_uncompressed_ptrs[i] = new_ptr;
      host_uncompressed_bytes[i] = new_bytes;

      host_oom_uncompressed_ptrs[n] = new_ptr;
      host_oom_uncompressed_bytes[n] = new_bytes;

      LOG_DEBUG(std::string("hipcompReallocateDecompressionBuffers: "
                            "reallocated buffer (#, old "
                            "#bytes, new #bytes): ") +
                std::to_string(i) + ", " + std::to_string(old_bytes) + ", " +
                std::to_string(new_bytes));
      n++;
    }
  }

  // allocate new device buffers
  const auto ptrs_bytes = sizeof(void *) * count;
  size_t bytes_bytes = count * sizeof(size_t);
  size_t oom_ptrs_bytes = sizeof(void *) * *oom_count;
  size_t oom_bytes_bytes = sizeof(size_t) * *oom_count;

  CHECK_HIP_API_CALL(
      hipMalloc((void **)device_oom_uncompressed_ptrs, oom_ptrs_bytes));
  CHECK_HIP_API_CALL(
      hipMalloc((void **)device_oom_uncompressed_bytes, oom_bytes_bytes));

  // copy host buffer arrays to device
  CHECK_HIP_API_CALL(hipMemcpyHtoD((void *)device_uncompressed_ptrs,
                                   (void *)host_uncompressed_ptrs.data(),
                                   ptrs_bytes));
  CHECK_HIP_API_CALL(hipMemcpyHtoD((void *)device_uncompressed_bytes,
                                   (void *)host_uncompressed_bytes.data(),
                                   bytes_bytes));

  CHECK_HIP_API_CALL(hipMemcpyHtoD((void *)*device_oom_uncompressed_ptrs,
                                   (void *)host_oom_uncompressed_ptrs.data(),
                                   oom_ptrs_bytes));
  CHECK_HIP_API_CALL(hipMemcpyHtoD((void *)*device_oom_uncompressed_bytes,
                                   (void *)host_oom_uncompressed_bytes.data(),
                                   oom_bytes_bytes));
  return hipcompSuccess;
}
