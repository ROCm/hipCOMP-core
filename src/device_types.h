// Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#ifndef DEVICE_TYPES_HIPH
#define DEVICE_TYPES_HIPH

#include <stdint.h>

namespace hipcomp {

#if !defined(USE_WARPSIZE_32)
# if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
#   define USE_WARPSIZE_64
#  endif
#endif

// Snappy GPU types
#if defined(USE_WARPSIZE_64)
typedef uint64_t warp_mask_t;
typedef int64_t signed_warp_mask_t;
#else
typedef uint32_t warp_mask_t;
typedef int32_t signed_warp_mask_t;
#endif

constexpr uint32_t uwarpsize = sizeof(warp_mask_t)*8;
constexpr int32_t warpsize = uwarpsize;

} // namespace hipcomp

#endif // DEVICE_TYPES_HIPH
