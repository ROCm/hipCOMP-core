<!---
    MIT License

    Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
-->

# Third-party tests

The tests in this folder are derived from third-party examples and tests.

## Requirements

The tests require the following ROCm and third-party libraries to be installed prior to compilation.

``deflate_cpu_compression_test.cpp`` and ``gzip_cpu_compression_test.cpp``:

* rocThrust
* libdeflate
* zlib

<!--, zlib, lz4, snappy, and zstd.-->

On Ubuntu, the external dependencies can be installed via the below command:

<!--
# LZ4
sudo apt-get install liblz4-dev
sudo apt-get install liblz4-1
---
# ZLib & Libdeflate
---
# Snappy
sudo apt-get install libsnappy-dev
sudo apt-get install libsnappy1v5
# Zstandard
sudo apt-get install libzstd-dev
sudo apt-get install libzstd1
-->

```sh
# ZLib
sudo apt-get install zlib1g-dev
sudo apt-get install zlib1g
# Libdeflate
sudo apt-get install libdeflate-dev
sudo apt-get install libdeflate0
```
