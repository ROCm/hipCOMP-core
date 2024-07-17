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

#include <vector>

#define CATCH_CONFIG_MAIN
#define HIPCOMP_DEBUG_OUTPUT 1

#include "Check.h"
#include "hip/hip_runtime_api.h"
#include "hipcomp/helpers.h"
#include "tests/catch.hpp"

/////////////////////////////////////////////////////////////

TEST_CASE("PadDeviceBufferArraySizesBytesOnHostTest", "[small]") {
  int count = 4;
  std::vector<size_t> bytes(count);
  int i = 0;
  bytes[i++] = 10;  // will be padded to size 128
  bytes[i++] = 128; // will be padded to size 128
  bytes[i++] = 129; // will be padded to size 256
  bytes[i++] = 932; // will be padded to size 1024
  REQUIRE(i == count);

  hipcompPadDeviceBufferArraySizes(
      bytes.data() /* const size_t* const bytes, array of size_t */,
      count /* int count */, 128 /* int padding_factor */,
      false /* bool bytes_on_device */);

  i = 0;
  REQUIRE(bytes[i++] == 128);
  REQUIRE(bytes[i++] == 128);
  REQUIRE(bytes[i++] == 256);
  REQUIRE(bytes[i++] == 1024);
  REQUIRE(i == count);
}

TEST_CASE("PadDeviceBufferArraySizesBytesOnDeviceTest", "[small]") {
  int count = 4;
  std::vector<size_t> bytes(count);
  int i = 0;
  bytes[i++] = 10;  // will be padded to size 128
  bytes[i++] = 128; // will be padded to size 128
  bytes[i++] = 129; // will be padded to size 256
  bytes[i++] = 932; // will be padded to size 1024
  REQUIRE(i == count);

  size_t *bytes_device = nullptr;
  size_t bytes_bytes = count * sizeof(size_t);

  CHECK_HIP_API_CALL(hipMalloc((void **)&bytes_device, bytes_bytes));
  CHECK_HIP_API_CALL(
      hipMemcpyHtoD((void *)bytes_device, (void *)bytes.data(), bytes_bytes));

  hipcompPadDeviceBufferArraySizes(
      bytes_device /* const size_t* const bytes_device, array of size_t */,
      count /* int count */, 128 /* int padding_factor */,
      true /* bool bytes_on_device */);

  CHECK_HIP_API_CALL(
      hipMemcpyDtoH((void *)bytes.data(), (void *)bytes_device, bytes_bytes));

  i = 0;
  REQUIRE(bytes[i++] == 128);
  REQUIRE(bytes[i++] == 128);
  REQUIRE(bytes[i++] == 256);
  REQUIRE(bytes[i++] == 1024);
  REQUIRE(i == count);

  CHECK_HIP_API_CALL(hipFree(bytes_device));
}

/////////////////////////////////////////////////////////////

TEST_CASE("AllocAndFreeBytesOnHostTest", "[small]") {
  void **ptrs = nullptr;

  int count = 4;
  std::vector<size_t> bytes(count);
  int i = 0;
  bytes[i++] = 10;  // will be padded to size 128
  bytes[i++] = 128; // will be padded to size 128
  bytes[i++] = 129; // will be padded to size 256
  bytes[i++] = 932; // will be padded to size 1024
  REQUIRE(i == count);

  hipcompAllocateDeviceBufferArray(
      &ptrs /* address of array of void* */,
      bytes.data() /* const size_t* const bytes, array of size_t */,
      count /* int count */, 128 /* int padding_factor */,
      false /* bool bytes_on_device */);

  i = 0;
  REQUIRE(bytes[i++] == 128);
  REQUIRE(bytes[i++] == 128);
  REQUIRE(bytes[i++] == 256);
  REQUIRE(bytes[i++] == 1024);
  REQUIRE(i == count);

  hipcompFreeDeviceBufferArray(ptrs, count);

  ptrs = nullptr;
}

TEST_CASE("AllocAndFreeBytesOnDeviceTest", "[small]") {
  void **ptrs = nullptr;

  int count = 4;
  std::vector<size_t> bytes(count);
  int i = 0;
  bytes[i++] = 10;  // will be padded to size 128
  bytes[i++] = 128; // will be padded to size 128
  bytes[i++] = 129; // will be padded to size 256
  bytes[i++] = 932; // will be padded to size 1024
  REQUIRE(i == count);

  size_t *bytes_device = nullptr;
  size_t bytes_bytes = count * sizeof(size_t);

  CHECK_HIP_API_CALL(hipMalloc((void **)&bytes_device, bytes_bytes));
  CHECK_HIP_API_CALL(
      hipMemcpyHtoD((void *)bytes_device, (void *)bytes.data(), bytes_bytes));

  hipcompAllocateDeviceBufferArray(
      &ptrs /* address of array of void* */,
      bytes_device /* const size_t* const bytes_device, array of size_t */,
      count /* int count */, 128 /* int padding_factor */,
      true /* bool bytes_on_device */);

  CHECK_HIP_API_CALL(
      hipMemcpyDtoH((void *)bytes.data(), (void *)bytes_device, bytes_bytes));

  i = 0;
  REQUIRE(bytes[i++] == 128);
  REQUIRE(bytes[i++] == 128);
  REQUIRE(bytes[i++] == 256);
  REQUIRE(bytes[i++] == 1024);
  REQUIRE(i == count);

  hipcompFreeDeviceBufferArray(ptrs, count);

  CHECK_HIP_API_CALL(hipFree(bytes_device));

  ptrs = nullptr;
}

/////////////////////////////////////////////////////////////

TEST_CASE("hipcompReallocateDecompressionBuffers", "[small]") {
  void **device_uncompressed_ptrs = nullptr;

  int count = 4;
  std::vector<size_t> bytes(count);
  size_t bytes_bytes = count * sizeof(size_t);
  int i = 0;
  bytes[i++] = 10;
  bytes[i++] = 128;
  bytes[i++] = 129;
  bytes[i++] = 932;
  REQUIRE(i == count);

  hipcompAllocateDeviceBufferArray(
      &device_uncompressed_ptrs /* address of array of void* */,
      bytes.data() /* const size_t* const bytes, array of size_t */,
      4 /* int count */, 0 /* NO PADDING, int padding_factor */,
      false /* bool bytes_on_device */);

  // estimated sizes
  size_t *device_uncompressed_bytes = nullptr;
  CHECK_HIP_API_CALL(hipMalloc(&device_uncompressed_bytes, bytes_bytes));
  CHECK_HIP_API_CALL(hipMemcpyHtoD((void *)device_uncompressed_bytes,
                                   (void *)bytes.data(), bytes_bytes));

  // actual sizes
  std::vector<size_t> bytes_new(count);
  i = 0;
  bytes_new[i++] = 128; // => out of memory
  bytes_new[i++] = 128;
  bytes_new[i++] = 256; // => out of memory
  bytes_new[i++] =
      1024; // => out of memory, oom_count == 3, oom indices: {0,2,3}

  size_t *device_actual_uncompressed_bytes = nullptr;
  CHECK_HIP_API_CALL(hipMalloc(&device_actual_uncompressed_bytes, bytes_bytes));
  CHECK_HIP_API_CALL(hipMemcpyHtoD((void *)device_actual_uncompressed_bytes,
                                   (void *)bytes_new.data(), bytes_bytes));

  // outputs
  int oom_count = 0;
  void **device_oom_uncompressed_ptrs = nullptr;
  size_t *device_oom_uncompressed_bytes = nullptr;

  hipcompReallocateDecompressionBuffersDebug(
      &oom_count,                    // int*
      &device_oom_uncompressed_ptrs, // void***            , pointer to array of
                                     // void*
      &device_oom_uncompressed_bytes,   // size_t**           , pointer to array
                                        // of size_t
      device_uncompressed_ptrs,         // void**             , array of void*
      device_uncompressed_bytes,        // size_t*            , array of size_t
      device_actual_uncompressed_bytes, // const size_t* const, array of size_t
      count,                            // int count
      0                                 // int padding_factor
  );
  REQUIRE(oom_count == 3);

  // check new number of bytes and ptrs
  std::vector<size_t> host_oom_uncompressed_bytes(oom_count);
  std::vector<void *> host_oom_uncompressed_ptrs(oom_count);
  std::vector<size_t> host_uncompressed_bytes_new(count);
  std::vector<void *> host_uncompressed_ptrs_new(count);
  CHECK_HIP_API_CALL(hipMemcpyDtoH((void *)host_oom_uncompressed_bytes.data(),
                                   (void *)device_oom_uncompressed_bytes,
                                   oom_count * sizeof(size_t)));
  CHECK_HIP_API_CALL(hipMemcpyDtoH((void *)host_oom_uncompressed_ptrs.data(),
                                   (void *)device_oom_uncompressed_ptrs,
                                   oom_count * sizeof(void *)));
  CHECK_HIP_API_CALL(hipMemcpyDtoH((void *)host_uncompressed_bytes_new.data(),
                                   (void *)device_uncompressed_bytes,
                                   count * sizeof(size_t)));
  CHECK_HIP_API_CALL(hipMemcpyDtoH((void *)host_uncompressed_ptrs_new.data(),
                                   (void *)device_uncompressed_ptrs,
                                   count * sizeof(void *)));

  i = 0;
  REQUIRE(host_oom_uncompressed_bytes[i++] == 128);
  REQUIRE(host_oom_uncompressed_bytes[i++] == 256);
  REQUIRE(host_oom_uncompressed_bytes[i++] == 1024);
  REQUIRE(i == oom_count);

  i = 0;
  REQUIRE(host_oom_uncompressed_bytes[i++] == host_uncompressed_bytes_new[0]);
  REQUIRE(host_oom_uncompressed_bytes[i++] == host_uncompressed_bytes_new[2]);
  REQUIRE(host_oom_uncompressed_bytes[i++] == host_uncompressed_bytes_new[3]);
  REQUIRE(i == oom_count);

  i = 0;
  REQUIRE(host_oom_uncompressed_ptrs[i++] == host_uncompressed_ptrs_new[0]);
  REQUIRE(host_oom_uncompressed_ptrs[i++] == host_uncompressed_ptrs_new[2]);
  REQUIRE(host_oom_uncompressed_ptrs[i++] == host_uncompressed_ptrs_new[3]);
  REQUIRE(i == oom_count);

  // clean up
  CHECK_HIP_API_CALL(hipFree(device_uncompressed_bytes));
  CHECK_HIP_API_CALL(hipFree(device_actual_uncompressed_bytes));
  CHECK_HIP_API_CALL(hipFree(device_oom_uncompressed_bytes));

  CHECK_API_CALL(hipcompFreeDeviceBufferArray(device_uncompressed_ptrs, count));
  // must not be freed this way as buffers are already freed by previous call:
  // CHECK_API_CALL(hipcompFreeDeviceBufferArray(device_oom_uncompressed_ptrs,
  // count));
  CHECK_HIP_API_CALL(hipFree(device_oom_uncompressed_ptrs));

  device_actual_uncompressed_bytes = nullptr;
  device_actual_uncompressed_bytes = nullptr;
  device_oom_uncompressed_bytes = nullptr;

  device_oom_uncompressed_ptrs = nullptr;
  device_uncompressed_ptrs = nullptr;
}
