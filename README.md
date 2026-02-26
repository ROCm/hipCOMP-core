# hipCOMP (v2.3.0)

> [!CAUTION]
> This release is an *early-access* software technology preview. Running
> production workloads is *not* recommended.

***

hipCOMP is a library for fast lossless compression/decompression on AMD GPUs.
The code is based on ``nvCOMP`` release branch
[``branch-2.2``](https://github.com/NVIDIA/nvcomp/tree/branch-2.2).

This release is an early-access technology preview focused on providing
GPU-accelerated compression for AMD hardware. AMD GPU implementations are
experimental and not yet performance-optimized.

## Algorithm Support

|  Algorithm   | Compression | Decompression | Optimized |               Status               |
| ------------ | ----------- | ------------- | --------- | ---------------------------------- |
| Snappy       | ✓           | ✓             | ✗        | Supported                          |
| LZ4          | ✓           | ✓             | ✗        | Supported                          |
| Cascaded     | ✓           | ✓             | ✗        | Supported                          |
| RLE          | ✓           | ✓             | ✗        | Supported                          |
| ZSTD         | ✗           | ✓             | ✗        | Experimental (decompress only)     |
| Deflate/GZIP | ✗           | ✓             | ✗        | Experimental (decompress only)     |
| Bitcomp      | ✗           | ✗             | ✗        | Not supported (NVIDIA proprietary) |
| ANS          | ✗           | ✗             | ✗        | Not supported (NVIDIA proprietary) |
| GDeflate     | ✗           | ✗             | ✗        | Not supported (NVIDIA proprietary) |

## Supported Platforms

* This release of hipCOMP has been successfully tested on AMD GPUs with
  ROCm 7.1.1, ROCm 7.2.0, TheRock 7.11.0 RC2.
* The following architectures have been tested successfully for this release:
  * gfx1030
  * gfx1100
  * gfx90a (MI210x)
  * gfx942 (MI300a/MI300x/MI308x)
* The library is built using CMake and the build requires a C++17 compliant
  compiler.
* The tests for GZIP/Deflate and ZSTD require the corresponding compression
  libraries to be installed on the system. See tests/third-party/CMakeLists.txt
  for details.
* The library supports both HIP/AMD and HIP/CUDA backends. The CUDA backend is
  available for testing and development purposes but is not the primary focus
  of this release.

## Build From Source

### HIP/AMD

> [!NOTE]
> If your system has no GPUs installed or you don't want to rely on automatic
> detection, use the `CMAKE_HIP_ARCHITECTURES` CMake option to set the GPU
> architecture(s) that you want to compile for. If you want to specify multiple
> architectures, use ';' to separate them.

```bash
cd hipcomp-core/
mkdir build/
cd build/
CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake cmake ../ -D CMAKE_HIP_ARCHITECTURES="gfxABC[; gfxBCD[; gfx...]]"
# To build with tests, append `-D BUILD_TESTS=1`:
# CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake cmake ../ -D BUILD_TESTS=1
make
```

### HIP/CUDA

> [!NOTE]
> If your system has no GPUs installed or you don't want to rely on automatic
> detection, use the `CMAKE_CUDA_ARCHITECTURES` CMake option to set the GPU
> architecture(s) that you want to compile for. If you want to specify multiple
> architectures, use ';' to separate them.

Like HIP/AMD but with additional `-D CUDA_BACKEND=1` option:

```bash
cd hipcomp-core/
mkdir build/
cd build/
CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake cmake ../ -D CUDA_BACKEND=1
# To build with tests, append `-D BUILD_TESTS=1`:
# CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake cmake ../ -D BUILD_TESTS=1 -D CUDA_BACKEND=1
make
```

#### Debugging

To create debug builds append the following option:

```bash
-D CMAKE_BUILD_TYPE=Debug
```

### Run tests

After completing the build, run:

```bash
cd hipcomp-core/
cd build/
make test
```

Tips:

* Select a particular GPU by setting the environment variable
  `HIP_VISIBLE_DEVICES=<id>` (or `CUDA_VISIBLE_DEVICES=<id>` with CUDA backend)
  before running ``make test``.
