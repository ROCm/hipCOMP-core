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
# libraries the tests link against and the CPack backends used by
# ci/internal/build.sh.
#
# No required environment variables.

set -euxo pipefail

os_id="$(. /etc/os-release; echo "${ID:-}" | tr "[:upper:]" "[:lower:]")"
os_id_like="$(. /etc/os-release; echo "${ID_LIKE:-}" | tr "[:upper:]" "[:lower:]")"

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
fi
