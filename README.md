# hipCOMP-CORE

hipCOMP CORE is a library for fast lossless compression/decompression on AMD MI series GPUs.

The code is based on ``nvCOMP`` release branch [``branch-2.2``](https://github.com/NVIDIA/nvcomp/tree/branch-2.2).
GPU implementations of algorithms like DEFLATE, GDEFLATE, ANS, and zSTD that have been later introduced as part of nvCOMP's proprietary
releases are not part of this repo (yet).

## Build From Source

### HIP/AMD

> **NOTE:** If you experience compiler errors related to ``cooperative_groups`` with ROCm versions ``<=6.0.X``, additionally specify the ``CMake`` build option `-D CG_WORKAROUND=1`.

```bash
cd hipcomp-core/
mkdir build/
cd build/
CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake cmake ../
# To build with tests, append `-D BUILD_TESTS=1`:
# CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake cmake ../ -D BUILD_TESTS=1
make
```

### HIP/CUDA

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

* Select a particular GPU by setting the environment variable `HIP_VISIBLE_DEVICES=<id>` (or `CUDA_VISIBLE_DEVICES=<id>` with CUDA backend) before running ``make test``.
