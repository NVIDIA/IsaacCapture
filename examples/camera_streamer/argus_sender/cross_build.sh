#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Cross-compile argus_sender for a Jetson on an x86 PC, using an Orin sysroot and an Arm GNU toolchain under
# ORIN_CROSS_ROOT (default ~/github/orin-cross-39.2.1):
#   <root>/sysroot                      the Jetson Linux root: the L4T packages (including nvidia-l4t-multimedia, which
#                                       holds libnvv4l2), the Multimedia API and, for the SIPL backend, the SIPL API
#   <root>/arm-gnu-toolchain-*/         Arm's aarch64-none-linux-gnu toolchain
#   <root>/orin-aarch64.cmake           a CMake toolchain file naming both
# `syntax` also needs clang, and the yaml-cpp sources that a first `build` fetches into <root>/build-streamer.
#
#   cross_build.sh syntax   type-check every source against the Orin headers (clang, no linking)
#   cross_build.sh build    configure and build the aarch64 argus_sender (extra arguments go to cmake)

set -euo pipefail

mode="${1:-build}"
shift || true
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${ORIN_CROSS_ROOT:-${HOME}/github/orin-cross-39.2.1}"
sysroot="${root}/sysroot"
mmapi="${sysroot}/usr/src/jetson_multimedia_api"
toolchain="${root}/orin-aarch64.cmake"

for path in "${sysroot}" "${mmapi}/argus/include" "${toolchain}" "${sysroot}/usr/lib/aarch64-linux-gnu/nvidia/libnvv4l2.so"; do
    if [[ ! -e "${path}" ]]; then
        echo "error: ${path} is missing; see the header of this script" >&2
        exit 1
    fi
done

case "${mode}" in
syntax)
    cxx_include="$(ls -d "${root}"/arm-gnu-toolchain*/aarch64-none-linux-gnu/include/c++/* | head -n1)"
    yaml_include="${root}/build-streamer/_deps/yaml_cpp-src/include"
    if [[ ! -d "${yaml_include}" ]]; then
        echo "error: ${yaml_include} is missing; run 'cross_build.sh build' once to fetch yaml-cpp" >&2
        exit 1
    fi
    status=0
    for source in "${project_dir}"/src/*.cpp; do
        # The SIPL headers need C++17; the rest of the sender is C++14.
        standard=c++14
        [[ "${source}" == */SiplCapture.cpp ]] && standard=c++17
        clang++ --target=aarch64-linux-gnu --sysroot="${sysroot}" -std="${standard}" -fsyntax-only -Wno-everything \
            -I"${sysroot}/usr/src/jetson_sipl_api/sipl/include" -I"${sysroot}/usr/src/jetson_sipl_api/sipl/include/nvsci" -I"${sysroot}/usr/src/jetson_sipl_api/sipl/include/query/include" \
            -DNVMMAPI_SUPPORTED -I"${project_dir}/src" -I"${mmapi}/argus/include" -I"${mmapi}/include" \
            -I"${mmapi}/samples/common/classes" -I/usr/local/cuda/targets/sbsa-linux/include \
            -I"${sysroot}/usr/include/drm" -I"${yaml_include}" -isystem "${sysroot}/usr/include/aarch64-linux-gnu" \
            -isystem "${cxx_include}" -isystem "${cxx_include}/aarch64-none-linux-gnu" \
            "${source}" || status=1
    done
    exit "${status}"
    ;;
build)
    build_dir="${root}/build-streamer"
    cmake -S "${project_dir}" -B "${build_dir}" -DCMAKE_TOOLCHAIN_FILE="${toolchain}" -DCMAKE_BUILD_TYPE=Release \
        -DDISABLE_MULTIPROCESS=ON -DMMAPI_ROOT="${mmapi}" -DDRM_INCLUDE_DIR="${sysroot}/usr/include/drm" \
        -DBUILD_TESTING=OFF -DSIPL_ROOT="${sysroot}/usr/src/jetson_sipl_api/sipl" "$@"
    cmake --build "${build_dir}" --target argus_sender -j "$(nproc)"
    echo "Built ${build_dir}/argus_sender (aarch64; not run)"
    ;;
*)
    echo "usage: cross_build.sh syntax|build [cmake args]" >&2
    exit 2
    ;;
esac
