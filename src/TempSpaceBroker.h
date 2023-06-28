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

#ifndef TEMPSPACEBROKER_H
#define TEMPSPACEBROKER_H

#include <cstddef>

namespace hipcomp
{

class TempSpaceBroker
{
public:
  /**
   * @brief Create a new temp space broker.
   *
   * @param space The start of the region of memory to use.
   * @param bytes The size of the region of memory.
   */
  TempSpaceBroker(void* space, size_t bytes);

  /**
   * @brief Reserve a chunk of temp space.
   *
   * @tparam T The type of object to reserve space for.
   * @param ptr The pointer to set to the reserved space.
   * @param num The number of objects to reserve space for.
   *
   * @throws An exception if there is no more space left.
   */
  template <typename T>
  void reserve(T** ptr, const size_t num)
  {
    *ptr = reinterpret_cast<T*>(reserve(alignof(T), num, sizeof(T)));
  }

  void reserve(void** ptr, const size_t num)
  {
    // Make sure we still align to 8 bytes
    *ptr = reserve(alignof(size_t), num, 1);
  }

  /**
   * @brief Get the number of bytes remaining in this temp space.
   *
   * @return The number of bytes.
   */
  size_t spaceLeft() const;

  /**
   * @brief Get the next available temp space. Its size is returned by
   * `spaceLeft()`.
   *
   * NOTE: This space does not get reserved, so any further calls to
   * `reserve()` may use this space.
   *
   * @return The next available space.
   */
  void* next() const;

private:
  void* m_base;
  size_t m_size;
  size_t m_offset;

  void* reserve(const size_t alignment, const size_t num, const size_t size);
};

} // namespace hipcomp

#endif