#!/usr/bin/env bash
# MIT License
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Run the hipCOMP-core CTest suite against a tree produced by
# ci/internal/build.sh. The tester images reach this script through the source
# tree shipped inside the tests tarball.
#
# Required environment variables:
#   BUILD_DIR  Scratch directory the build tree was created in.
#
# Optional environment variables:
#   PROJECT_ID              Checkout directory name (default: hipcomp_core).
#   HIPCOMP_CORE_CONDA_ENV  Conda environment to activate first.

set -xeu

: "${BUILD_DIR:?BUILD_DIR must be set}"

PROJECT_ID="${PROJECT_ID:-hipcomp_core}"
build_build_dir="${BUILD_DIR}/${PROJECT_ID}/build"

if [ -n "${HIPCOMP_CORE_CONDA_ENV:-}" ]; then
  set +u # note: conda script may have unbound variables
  source ${CONDA_DIR}/etc/profile.d/conda.sh
  conda activate ${HIPCOMP_CORE_CONDA_ENV}
  set -u
fi

# build.sh only creates this venv when the caller pins CMAKE_VERSION; it is
# where ctest lives in that case.
if [ -x "${BUILD_DIR}/${PROJECT_ID}/_venv/bin/activate" ] ||
   [ -f "${BUILD_DIR}/${PROJECT_ID}/_venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  . "${BUILD_DIR}/${PROJECT_ID}/_venv/bin/activate"
fi

ctest -V --test-dir "${build_build_dir}"
