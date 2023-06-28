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

#include "CompressionConfigs.hpp"

namespace hipcomp {

CompressionConfig::CompressionConfigImpl::CompressionConfigImpl(PinnedPtrPool<hipcompStatus_t>& pool)
  : status(pool.allocate())
{
  *get_status() = hipcompSuccess;
}

hipcompStatus_t* CompressionConfig::CompressionConfigImpl::get_status() const {
  return status->get_ptr();
}

CompressionConfig::CompressionConfig(PinnedPtrPool<hipcompStatus_t>& pool, size_t uncompressed_buffer_size)
  : impl(std::make_shared<CompressionConfig::CompressionConfigImpl>(pool)),
    uncompressed_buffer_size(uncompressed_buffer_size),
    max_compressed_buffer_size(0),
    num_chunks(0)
{}

hipcompStatus_t* CompressionConfig::get_status() const {
  return impl->get_status();
}

CompressionConfig::~CompressionConfig() {}

CompressionConfig::CompressionConfig(CompressionConfig&& other)
  : impl(std::move(other.impl)),
    uncompressed_buffer_size(other.uncompressed_buffer_size),
    max_compressed_buffer_size(other.max_compressed_buffer_size),
    num_chunks(other.num_chunks)
{}

CompressionConfig::CompressionConfig(const CompressionConfig& other)
  : impl(other.impl),
    uncompressed_buffer_size(other.uncompressed_buffer_size),
    max_compressed_buffer_size(other.max_compressed_buffer_size),
    num_chunks(other.num_chunks)
{}

CompressionConfig& CompressionConfig::operator=(const CompressionConfig& other) 
{
  impl = other.impl;
  uncompressed_buffer_size = other.uncompressed_buffer_size;
  max_compressed_buffer_size = other.max_compressed_buffer_size;
  num_chunks = other.num_chunks;
  return *this;
}

CompressionConfig& CompressionConfig::operator=(CompressionConfig&& other) 
{
  impl = std::move(other.impl);
  uncompressed_buffer_size = other.uncompressed_buffer_size;
  max_compressed_buffer_size = other.max_compressed_buffer_size;
  num_chunks = other.num_chunks;
  return *this;
}

/**
 * @brief Construct the config given an hipcompStatus_t memory pool
 */
DecompressionConfig::DecompressionConfigImpl::DecompressionConfigImpl(PinnedPtrPool<hipcompStatus_t>& pool)
  : status(pool.allocate()),
    decomp_data_size(),
    num_chunks()
{
  *get_status() = hipcompSuccess;
}

/**
 * @brief Get the raw hipcompStatus_t*
 */
hipcompStatus_t* DecompressionConfig::DecompressionConfigImpl::get_status() const {
  return status->get_ptr();
}

DecompressionConfig::DecompressionConfig(PinnedPtrPool<hipcompStatus_t>& pool)
  : impl(std::make_shared<DecompressionConfig::DecompressionConfigImpl>(pool)),
    decomp_data_size(0),
    num_chunks(0)
{}

hipcompStatus_t* DecompressionConfig::get_status() const {
  return impl->get_status();
}

DecompressionConfig::~DecompressionConfig() {}

DecompressionConfig::DecompressionConfig(DecompressionConfig&& other)
  : impl(std::move(other.impl)),
    decomp_data_size(other.decomp_data_size),
    num_chunks(other.num_chunks)
{}

DecompressionConfig& DecompressionConfig::operator=(const DecompressionConfig& other) 
{
  impl = other.impl;
  decomp_data_size = other.decomp_data_size;
  num_chunks = other.num_chunks;
  return *this;
}

DecompressionConfig& DecompressionConfig::operator=(DecompressionConfig&& other) 
{
  impl = std::move(other.impl);
  decomp_data_size = other.decomp_data_size;
  num_chunks = other.num_chunks;
  return *this;
}

DecompressionConfig::DecompressionConfig(const DecompressionConfig& other)
  : impl(other.impl),
    decomp_data_size(other.decomp_data_size),
    num_chunks(other.num_chunks)
{}

} // namespace hipcomp