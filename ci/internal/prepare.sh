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

# Install OS-level prerequisites for hipCOMP-core CI builds: the compression
# libraries the tests link against, the ROCm dev headers the build includes,
# and the CPack backends used by ci/internal/build.sh.
#
# No required environment variables.
#
# Optional environment variables:
#   ROCM_PATH  ROCm root inspected for headers that are already present
#              (default: /opt/rocm).

set -euxo pipefail

rocm_root="${ROCM_PATH:-/opt/rocm}"

os_id="$(. /etc/os-release; echo "${ID:-}" | tr "[:upper:]" "[:lower:]")"
os_id_like="$(. /etc/os-release; echo "${ID_LIKE:-}" | tr "[:upper:]" "[:lower:]")"

# ROCm dev headers as "<include subdirectory> <Debian package> <RHEL package>".
# A ROCm stack installed from pip wheels ships these headers inside its own
# tree and has no distro package for them, so whether the headers are already
# in place decides whether a package is wanted at all.
rocm_dev_packages=(
    "hipcub hipcub-dev hipcub-devel"
    "thrust rocthrust-dev rocthrust-devel"
)

debian_rocm_packages=()
rhel_rocm_packages=()
for entry in "${rocm_dev_packages[@]}"; do
    read -r include_dir debian_package rhel_package <<< "${entry}"
    if [[ -d "${rocm_root}/include/${include_dir}" ]]; then
        echo "prepare.sh: ${rocm_root}/include/${include_dir} exists," \
             "not installing ${debian_package}/${rhel_package}"
        continue
    fi
    debian_rocm_packages+=("${debian_package}")
    rhel_rocm_packages+=("${rhel_package}")
done

if [[ "${os_id}" == *"ubuntu"* ]] || [[ "${os_id}" == *"debian"* ]] || [[ "${os_id_like}" == *"debian"* ]]; then
    APT_CMD="apt-get"
    command -v sudo >/dev/null 2>&1 && APT_CMD="sudo apt-get"

    packages=(
        zlib1g zlib1g-dev
        libdeflate0 libdeflate-dev
        libzstd1 libzstd-dev
    )
    # build.sh needs at least one CPack backend.
    if ! command -v rpmbuild >/dev/null 2>&1; then
        packages+=("rpm")
    fi
    if ! command -v dpkg-deb >/dev/null 2>&1; then
        packages+=("dpkg-dev")
    fi

    ${APT_CMD} update
    ${APT_CMD} install -y --no-install-recommends "${packages[@]}"

    # Best effort, one package at a time so that a name missing from the
    # repositories cannot hide one that is available. A build against a ROCm
    # tree that turns out not to carry the headers either fails later in
    # find_package(), which reports the missing dependency far more precisely.
    for package in ${debian_rocm_packages[@]+"${debian_rocm_packages[@]}"}; do
        ${APT_CMD} install -y --no-install-recommends "${package}" ||
            echo "prepare.sh: ${package} is not installable," \
                 "expecting its headers in ${rocm_root}"
    done
elif [[ "${os_id}" == *"rhel"* ]] || [[ "${os_id_like}" == *"rhel"* ]] || [[ "${os_id_like}" == *"fedora"* ]]; then
    DNF_CMD="dnf"
    command -v dnf >/dev/null 2>&1 || DNF_CMD="yum"
    command -v sudo >/dev/null 2>&1 && DNF_CMD="sudo ${DNF_CMD}"

    packages=(
        zlib-devel
        libdeflate-devel
        libzstd-devel
    )
    # manylinux/AlmaLinux images have no dpkg, so only the RPM backend is set up.
    if ! command -v rpmbuild >/dev/null 2>&1; then
        packages+=("rpm-build")
    fi

    ${DNF_CMD} install -y "${packages[@]}"

    for package in ${rhel_rocm_packages[@]+"${rhel_rocm_packages[@]}"}; do
        ${DNF_CMD} install -y --nogpgcheck "${package}" ||
            echo "prepare.sh: ${package} is not installable," \
                 "expecting its headers in ${rocm_root}"
    done
fi
