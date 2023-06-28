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
#include "TempSpaceBroker.h"

#include "hip/hip_runtime.h"

#include <cstdint>

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

struct Test32BStruct
{
  uint8_t data[32];
};

template <typename T>
void checked_alloc(TempSpaceBroker& temp, const size_t num)
{
  const size_t size = temp.spaceLeft();
  const void* ptr = temp.next();

  T* first;
  temp.reserve(&first, num);

  // it may have rounded up to get alignment
  REQUIRE(temp.spaceLeft() <= size - sizeof(T) * num);
  // but don't let it round up more than seven bytes on current platforms
  REQUIRE(temp.spaceLeft() + 7 >= size - sizeof(T) * num);

  // make sure we moved the next available allocation
  REQUIRE(temp.next() > ptr);

  // make sure the size removed at least fits our allocation
  const size_t move_size = static_cast<size_t>(
      static_cast<const uint8_t*>(temp.next())
      - static_cast<const uint8_t*>(ptr));
  REQUIRE(move_size >= sizeof(T) * num);
}

template <typename T>
void test_base_alloc(const size_t size, const size_t num)
{
  void* ptr;
  HIP_RT_CALL(hipMalloc(&ptr, size));

  TempSpaceBroker temp(ptr, size);

  checked_alloc<T>(temp, num);

  hipFree(ptr);
}

template <typename T>
void test_base_alloc_exception(const size_t size, const size_t num)
{
  void* ptr;
  HIP_RT_CALL(hipMalloc(&ptr, size));

  TempSpaceBroker temp(ptr, size);

  try {
    T* first;
    temp.reserve(&first, num);
    REQUIRE(false);
  } catch (const std::exception&) {
    // pass
  }

  hipFree(ptr);
}

/******************************************************************************
 * UNIT TEST ******************************************************************
 *****************************************************************************/

TEST_CASE("MixedSizeTest", "[small]")
{
  void* ptr;
  const size_t size = 1024;
  HIP_RT_CALL(hipMalloc(&ptr, size));

  TempSpaceBroker temp(ptr, size);

  checked_alloc<int16_t>(temp, 5);
  checked_alloc<double*>(temp, 1);
  checked_alloc<double>(temp, 7);
  checked_alloc<char>(temp, 1);
  checked_alloc<int32_t>(temp, 25);
  checked_alloc<Test32BStruct>(temp, 3);
  checked_alloc<double>(temp, 7);

  hipFree(ptr);
}

TEST_CASE("AllBaseTypeTest", "[small]")
{
  test_base_alloc<int8_t>(1000, 31);
  test_base_alloc<uint8_t>(1000, 31);
  test_base_alloc<int16_t>(1000, 31);
  test_base_alloc<uint16_t>(1000, 31);
  test_base_alloc<int32_t>(1000, 31);
  test_base_alloc<uint32_t>(1000, 31);
  test_base_alloc<int64_t>(1000, 31);
  test_base_alloc<uint64_t>(1000, 31);
}

TEST_CASE("AllBaseTypeExactSizeTest", "[small]")
{
  test_base_alloc<int8_t>(1024, 1024);
  test_base_alloc<uint8_t>(1024, 1024);
  test_base_alloc<int16_t>(1024, 512);
  test_base_alloc<uint16_t>(1024, 512);
  test_base_alloc<int32_t>(1024, 256);
  test_base_alloc<uint32_t>(1024, 256);
  test_base_alloc<int64_t>(1024, 128);
  test_base_alloc<uint64_t>(1024, 128);
}

TEST_CASE("AllBaseTypeOverflowTest", "[small]")
{
  test_base_alloc_exception<int8_t>(1023, 1024);
  test_base_alloc_exception<uint8_t>(1023, 1024);
  test_base_alloc_exception<int16_t>(1023, 512);
  test_base_alloc_exception<uint16_t>(1023, 512);
  test_base_alloc_exception<int32_t>(1023, 256);
  test_base_alloc_exception<uint32_t>(1023, 256);
  test_base_alloc_exception<int64_t>(1023, 128);
  test_base_alloc_exception<uint64_t>(1023, 128);
}

TEST_CASE("Struct32BTest", "[small]")
{
  test_base_alloc<Test32BStruct>(10000, 19);
}