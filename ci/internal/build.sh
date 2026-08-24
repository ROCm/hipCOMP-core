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

# Build hipCOMP-core: the C++ library, its DEB/RPM packages and the tests
# tarball that the tester images consume.
#
# Required environment variables:
#   SRC_DIR              Parent directory of the project checkout.
#   BUILD_DIR            Scratch directory for the build tree.
#   INSTALL_DIR          Prefix the install step targets and the packages are
#                        built for.
#   BUILD_ARTIFACTS_DIR  Destination for packages and the tests tarball.
#
# Optional environment variables:
#   PROJECT_ID                       Checkout directory name under SRC_DIR
#                                    (default: hipcomp_core).
#   HIPCOMP_CORE_BUILD_TYPE          build | install | debug_build
#                                    (default: build).
#   HIPCOMP_CORE_CONDA_ENV           Conda environment to activate first.
#   HIPCOMP_CORE_INSTALL_COMPONENTS  Components installed when the build type is
#                                    'install' (default: devel; empty installs
#                                    everything).
#   HIPCOMP_CORE_CPACK_GENERATORS    Semicolon separated CPack generator list.
#                                    Defaults to the generators whose backend is
#                                    present (RPM and/or DEB).
#   HIPCOMP_CORE_CREATE_TESTS_TARBALL
#                                    '0' skips packing the tests tarball the
#                                    tester images consume (default: 1).
#   CMAKE_INSTALL_PREFIX             Install prefix, overriding INSTALL_DIR; the
#                                    literal '<conda-prefix>' resolves to
#                                    CONDA_PREFIX.
#   CPACK_PACKAGING_INSTALL_PREFIX   Prefix baked into the packages. Defaults to
#                                    the install prefix; set it explicitly to
#                                    package for a prefix other than the one
#                                    built against.
#   AMDGPU_TARGETS                   GPU architectures to compile for.
#   SCCACHE_ENABLE                   'true' enables the sccache launchers.
#   CMAKE_VERSION                    When set, cmake of that version is
#                                    installed into a venv under BUILD_DIR.
#   ROCM_PATH, ROCM_CMAKE_PATH, CMAKE_PREFIX_PATH
#   CMAKE_BUILD_PARALLEL_LEVEL, MAX_JOBS   Build parallelism.

set -xeu

: "${SRC_DIR:?SRC_DIR must be set}"
: "${BUILD_DIR:?BUILD_DIR must be set}"
: "${INSTALL_DIR:?INSTALL_DIR must be set}"
: "${BUILD_ARTIFACTS_DIR:?BUILD_ARTIFACTS_DIR must be set}"

PROJECT_ID="${PROJECT_ID:-hipcomp_core}"

[[ "${HIPCOMP_CORE_BUILD_TYPE:-build}" == "install" ]] && do_install="1"
[[ "${HIPCOMP_CORE_BUILD_TYPE:-build}" == "debug_build" ]] && debug_mode="1"

if [ -n "${HIPCOMP_CORE_CONDA_ENV:-}" ]; then
  set +u # note: conda script may have unbound variables
  source ${CONDA_DIR}/etc/profile.d/conda.sh
  conda activate ${HIPCOMP_CORE_CONDA_ENV}
  set -u
fi

components="${HIPCOMP_CORE_INSTALL_COMPONENTS-devel}"
tests_tarball_name="${PROJECT_ID}-tests.tar.gz"

case "${HIPCOMP_CORE_CREATE_TESTS_TARBALL:-1}" in
  0 | false | no | off) create_tests_tarball="" ;;
  *) create_tests_tarball="1" ;;
esac

[[ "${CMAKE_INSTALL_PREFIX:-}" == "<conda-prefix>" ]] && install_to_conda_prefix="1"
[[ "${SCCACHE_ENABLE:-false}" == "true" ]] && use_sccache=1

if [ -d "/opt/rh/gcc-toolset-$(g++ -dumpversion)" ]; then
  toolchain="/opt/rh/gcc-toolset-$(g++ -dumpversion)/root/usr"
  export CCC_OVERRIDE_OPTIONS="+--gcc-toolchain=${toolchain}"
  # note: in some shells, cmake is using /bin/{cc, c++} as compilers,
  # which may not be the compilers of the enabled toolchain.
  export CC=${toolchain}/bin/cc
  export CXX=${toolchain}/bin/c++
fi

#### packaging backends

if [ -n "${HIPCOMP_CORE_CPACK_GENERATORS:-}" ]; then
  cpack_generators="${HIPCOMP_CORE_CPACK_GENERATORS}"
else
  detected=()
  if command -v rpmbuild >/dev/null 2>&1; then
    detected+=("RPM")
  fi
  if command -v dpkg-deb >/dev/null 2>&1; then
    detected+=("DEB")
  fi
  cpack_generators="$(IFS=';'; echo "${detected[*]:-}")"
fi

if [ -z "${cpack_generators}" ]; then
  echo "ERROR: no CPack backend available (need rpmbuild and/or dpkg-deb);" \
       "run ci/internal/prepare.sh first" >&2
  exit 2
fi

#### configure

cmake_args=(
  "-D BUILD_TESTS=1"
  "-D CMAKE_VERBOSE_MAKEFILE=ON"
  "-D CPACK_OUTPUT_FILE_PREFIX=${BUILD_ARTIFACTS_DIR}"
  "-D CPACK_GENERATOR=${cpack_generators}"
  ${debug_mode:+"-D CMAKE_BUILD_TYPE=Debug"}
  ${CMAKE_PREFIX_PATH:+"-D CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}${ROCM_PATH:+:${ROCM_PATH}/lib/cmake}"}
  ${ROCM_PATH:+"-D ROCM_PATH=${ROCM_PATH}"}
  ${ROCM_CMAKE_PATH:+"-D ROCM_CMAKE_PATH=${ROCM_CMAKE_PATH}"}
  ${AMDGPU_TARGETS:+"-D CMAKE_HIP_ARCHITECTURES=${AMDGPU_TARGETS}"}
  ${AMDGPU_TARGETS:+"-D GPU_TARGETS=${AMDGPU_TARGETS}"}
  ${use_sccache:+"-D CMAKE_HIP_COMPILER_LAUNCHER=sccache"}
  ${use_sccache:+"-D CMAKE_CXX_COMPILER_LAUNCHER=sccache"}
  ${use_sccache:+"-D CMAKE_C_COMPILER_LAUNCHER=sccache"}
)

# The packaging prefix follows the install prefix unless the caller pins it,
# which lets the Docker builders install into a staging root while still
# packaging for the ROCm prefix.
if [ -n "${install_to_conda_prefix:-}" ]; then
  cmake_args+=(
    "-D CMAKE_INSTALL_PREFIX=${CONDA_PREFIX}"
    "-D CPACK_PACKAGING_INSTALL_PREFIX=${CONDA_PREFIX}"
  )
else
  install_prefix="${CMAKE_INSTALL_PREFIX:-${INSTALL_DIR}}"
  package_prefix="${CPACK_PACKAGING_INSTALL_PREFIX:-${install_prefix}}"
  cmake_args+=(
    "-D CMAKE_INSTALL_PREFIX=${install_prefix}"
    "-D CPACK_PACKAGING_INSTALL_PREFIX=${package_prefix}"
  )
fi

build_args=(
  "--parallel ${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS:-1}}"
)

src_dir=${SRC_DIR}/${PROJECT_ID}
build_src_dir=${BUILD_DIR}/${PROJECT_ID}
build_build_dir=${BUILD_DIR}/${PROJECT_ID}/build

mkdir -p ${build_src_dir}
rm -rf ${build_src_dir} # NOTE: not necessary but helpful for debugging
cp -R ${src_dir} ${build_src_dir}

mkdir -p ${BUILD_ARTIFACTS_DIR}

# The pipeline pins cmake per build; the Docker images ship their own.
if [ -n "${CMAKE_VERSION:-}" ]; then
  python3 -m venv ${build_src_dir}/_venv
  . ${build_src_dir}/_venv/bin/activate
  pip3 install --upgrade pip
  pip3 install cmake==${CMAKE_VERSION}
fi

#### build, package, install

cmake -S ${build_src_dir} -B ${build_build_dir} ${cmake_args[@]}
cmake --build ${build_build_dir} ${build_args[@]}
cmake --build ${build_build_dir} --target package ${build_args[@]}

if [ -n "${do_install:-}" ]; then
  if [ -n "${components}" ]; then
    for component in ${components}; do
      cmake --install ${build_build_dir} --component ${component} --verbose
    done
  else
    cmake --install ${build_build_dir} --verbose
  fi
fi

#### test archive

# The tester images extract this at / and run ctest against ${build_build_dir},
# so the archive keeps absolute paths. The source tree comes along because the
# tester drives ci/internal/prepare.sh and ci/internal/test.sh out of it; .git
# would otherwise dominate the archive.
if [ -n "${create_tests_tarball:-}" ]; then
  tar --exclude=.git -czvf ${BUILD_ARTIFACTS_DIR}/${tests_tarball_name} ${build_build_dir} ${src_dir}
  du -sh ${BUILD_ARTIFACTS_DIR}/${tests_tarball_name}
fi

# inspect package contents
if command -v dpkg >/dev/null 2>&1; then
  find ${BUILD_ARTIFACTS_DIR} -name "*.deb" -exec echo "{}:" \; -exec dpkg -c {} \;
fi
