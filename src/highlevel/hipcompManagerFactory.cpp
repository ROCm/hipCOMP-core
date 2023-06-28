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

#include <assert.h>

#include "hipcomp.hpp"
#include "hipcomp/hipcompManager.hpp"
#include "hipcomp/ans.hpp"
#include "hipcomp/lz4.hpp"
#include "hipcomp/snappy.hpp"
#include "hipcomp/gdeflate.hpp"
#include "hipcomp/cascaded.hpp"
#include "hipcomp/bitcomp.hpp"
#include "hipcomp_common_deps/hlif_shared_types.hpp"
#include "HipUtils.h"

namespace hipcomp {

std::shared_ptr<hipcompManagerBase> create_manager(const uint8_t* comp_buffer, hipStream_t stream = 0, const int device_id = 0) {
  // Need to determine the type of manager
  const CommonHeader* common_header = reinterpret_cast<const CommonHeader*>(comp_buffer);
  CommonHeader cpu_common_header;
  HipUtils::check(hipMemcpyAsync(&cpu_common_header, common_header, sizeof(CommonHeader), hipMemcpyDefault, stream));
  HipUtils::check(hipStreamSynchronize(stream));

  std::shared_ptr<hipcompManagerBase> res;

  switch(cpu_common_header.format) {
    case FormatType::LZ4: 
    {
      LZ4FormatSpecHeader format_spec;
      const LZ4FormatSpecHeader* gpu_format_header = reinterpret_cast<const LZ4FormatSpecHeader*>(comp_buffer + sizeof(CommonHeader));
      HipUtils::check(hipMemcpyAsync(&format_spec, gpu_format_header, sizeof(LZ4FormatSpecHeader), hipMemcpyDefault, stream));
      HipUtils::check(hipStreamSynchronize(stream));

      res = std::make_shared<LZ4Manager>(cpu_common_header.uncomp_chunk_size, format_spec.data_type, stream, device_id);
      break;
    }
    case FormatType::Snappy: 
    {
      SnappyFormatSpecHeader format_spec;
      const SnappyFormatSpecHeader* gpu_format_header = reinterpret_cast<const SnappyFormatSpecHeader*>(comp_buffer + sizeof(CommonHeader));
      HipUtils::check(hipMemcpyAsync(&format_spec, gpu_format_header, sizeof(SnappyFormatSpecHeader), hipMemcpyDefault, stream));
      HipUtils::check(hipStreamSynchronize(stream));
      
      res = std::make_shared<SnappyManager>(cpu_common_header.uncomp_chunk_size, stream, device_id);
      break;
    }
    case FormatType::GDeflate: 
    {
      hipcompBatchedGdeflateOpts_t format_spec;
      const hipcompBatchedGdeflateOpts_t* gpu_format_header = reinterpret_cast<const hipcompBatchedGdeflateOpts_t*>(comp_buffer + sizeof(CommonHeader));
      HipUtils::check(hipMemcpyAsync(&format_spec, gpu_format_header, sizeof(hipcompBatchedGdeflateOpts_t), hipMemcpyDefault, stream));
      HipUtils::check(hipStreamSynchronize(stream));

      res = std::make_shared<GdeflateManager>(cpu_common_header.uncomp_chunk_size, format_spec.algo, stream, device_id);
      break;
    }
    case FormatType::Bitcomp: 
    {
#ifdef ENABLE_BITCOMP
      BitcompFormatSpecHeader format_spec;
      const BitcompFormatSpecHeader* gpu_format_header = reinterpret_cast<const BitcompFormatSpecHeader*>(comp_buffer + sizeof(CommonHeader));
      HipUtils::check(hipMemcpyAsync(&format_spec, gpu_format_header, sizeof(BitcompFormatSpecHeader), hipMemcpyDefault, stream));
      HipUtils::check(hipStreamSynchronize(stream));

      res = std::make_shared<BitcompManager>(format_spec.data_type, format_spec.algo, stream, device_id);
#else
      throw HipCompException(hipcompErrorNotSupported, "Bitcomp support not available in this build.");
#endif
      break;
    }
    case FormatType::ANS: 
    {
      ANSFormatSpecHeader format_spec;
      const ANSFormatSpecHeader* gpu_format_header = reinterpret_cast<const ANSFormatSpecHeader*>(comp_buffer + sizeof(CommonHeader));
      HipUtils::check(hipMemcpyAsync(&format_spec, gpu_format_header, sizeof(ANSFormatSpecHeader), hipMemcpyDefault, stream));
      HipUtils::check(hipStreamSynchronize(stream));

      res = std::make_shared<ANSManager>(cpu_common_header.uncomp_chunk_size, stream, device_id);
      break;
    }
    case FormatType::Cascaded: 
    {
      CascadedFormatSpecHeader format_spec;
      const CascadedFormatSpecHeader* gpu_format_header = reinterpret_cast<const CascadedFormatSpecHeader*>(comp_buffer + sizeof(CommonHeader));
      HipUtils::check(hipMemcpyAsync(&format_spec, gpu_format_header, sizeof(CascadedFormatSpecHeader), hipMemcpyDefault, stream));
      HipUtils::check(hipStreamSynchronize(stream));

      assert(cpu_common_header.uncomp_chunk_size == format_spec.options.chunk_size);

      res = std::make_shared<CascadedManager>(format_spec.options, stream, device_id);
      break;
    }
    case FormatType::NotSupportedError:
    {
      assert(false);
    }
  }

  return res;
}

} // namespace hipcomp 
 