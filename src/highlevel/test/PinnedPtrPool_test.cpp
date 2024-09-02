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

#define CATCH_CONFIG_MAIN

#include <vector>
#include "hip/hip_runtime.h"

#include "tests/catch.hpp"
#include "common.h"

#include "highlevel/PinnedPtrs.hpp"

using namespace hipcomp;
using namespace std;

namespace hipcomp {

template<typename T>
struct PoolTestWrapper {
  PinnedPtrPool<T>& pool;
  PoolTestWrapper(PinnedPtrPool<T>& pool) 
    : pool(pool)
  {}

  size_t get_current_available_pointer_count() {
    return pool.get_current_available_pointer_count();
  }

  size_t capacity() {
    return pool.capacity();
  }   
};

}

template<typename T>
void test_pinned_ptr_pool() {
  typedef PinnedPtrPool<T> PinnedPool;
  PinnedPool pool{};
  typedef std::unique_ptr<typename PinnedPool::PinnedPtrHandle> PinnedPtr;
  PoolTestWrapper<T> test_wrapper{pool};

  constexpr size_t num_pinned_prealloc = PINNED_POOL_PREALLOC_SIZE;
  constexpr size_t num_pinned_realloc = PINNED_POOL_REALLOC_SIZE;
  REQUIRE(test_wrapper.capacity() == num_pinned_prealloc);
  REQUIRE(test_wrapper.get_current_available_pointer_count() == num_pinned_prealloc);

  vector<PinnedPtr> pinned_ptrs;
  for (size_t i = 1; i <= num_pinned_prealloc; ++i)
  {
    pinned_ptrs.push_back(pool.allocate());
    
    REQUIRE(test_wrapper.get_current_available_pointer_count() == num_pinned_prealloc - i);
    REQUIRE(test_wrapper.capacity() == num_pinned_prealloc);
    
    **pinned_ptrs.back() = i;
  }

  REQUIRE(test_wrapper.capacity() == num_pinned_prealloc);
  REQUIRE(test_wrapper.get_current_available_pointer_count() == 0);

  pinned_ptrs.pop_back(); // return one to the pool
  REQUIRE(test_wrapper.capacity() == num_pinned_prealloc);
  REQUIRE(test_wrapper.get_current_available_pointer_count() == 1);

  for (size_t i = 0; i < 2; ++i) {
    pinned_ptrs.push_back(pool.allocate());
  }

  REQUIRE(test_wrapper.get_current_available_pointer_count() == num_pinned_realloc - 1);
  REQUIRE(test_wrapper.capacity() == num_pinned_realloc + num_pinned_prealloc);

  pinned_ptrs.clear();
  REQUIRE(test_wrapper.capacity() == num_pinned_realloc + num_pinned_prealloc);
  REQUIRE(test_wrapper.get_current_available_pointer_count() == num_pinned_realloc + num_pinned_prealloc);

}

TEST_CASE("test_pinned_ptr_pool_int")
{
  test_pinned_ptr_pool<int>();
}

TEST_CASE("test_pinned_ptr_pool_short")
{
  test_pinned_ptr_pool<short>();
}
