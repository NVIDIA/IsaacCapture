# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Camera-panel development dependencies stay inside this build directory.
# Keep the GPU driver/Vulkan runtime a platform prerequisite; never install host packages here.
include(FetchContent)
if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()

if(NOT Vulkan_FOUND)
    find_library(ISAAC_TELEOP_VULKAN_LOADER NAMES vulkan libvulkan.so.1)
    if(NOT ISAAC_TELEOP_VULKAN_LOADER)
        message(FATAL_ERROR "OVXR camera quads require the platform Vulkan runtime (libvulkan.so.1)")
    endif()
    # The pinned include directory has no CMakeLists.txt: download headers only,
    # without adding the vendor project, targets or install rules (CMake >=3.18).
    FetchContent_Declare(isaacteleop_vulkan_headers
        URL https://codeload.github.com/KhronosGroup/Vulkan-Headers/tar.gz/217e93c664ec6704ec2d8c36fa116c1a4a1e2d40
        URL_HASH SHA256=44650ba0438084df8c90f9601c73fb0e7a746ba187292d65d3ecf7f642c6112a
        SOURCE_SUBDIR include)
    FetchContent_MakeAvailable(isaacteleop_vulkan_headers)
    set(Vulkan_INCLUDE_DIR "${isaacteleop_vulkan_headers_SOURCE_DIR}/include")
    set(Vulkan_LIBRARY "${ISAAC_TELEOP_VULKAN_LOADER}")
    add_library(Vulkan::Vulkan UNKNOWN IMPORTED)
    set_target_properties(Vulkan::Vulkan PROPERTIES
        IMPORTED_LOCATION "${Vulkan_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Vulkan_INCLUDE_DIR}")
    set(Vulkan_FOUND TRUE)
    message(STATUS "OVXR camera quads: build-managed Vulkan headers with ${Vulkan_LIBRARY}")
endif()

if(NOT CUDAToolkit_FOUND)
    if(CUDAToolkit_ROOT OR DEFINED ENV{CUDAToolkit_ROOT})
        message(FATAL_ERROR "The specified CUDAToolkit_ROOT was not found; correct it or leave it unset for build-managed PiP SDK acquisition")
    endif()
    FetchContent_Declare(isaacteleop_cuda_cudart
        URL https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.0.107-archive.tar.xz
        URL_HASH SHA256=c0e16e9593a69afe4997ae96a8b024e751ebce0254523fd969b08bb028582d23)
    FetchContent_Declare(isaacteleop_cuda_nvcc
        URL https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.0.76-archive.tar.xz
        URL_HASH SHA256=160ca9f8828f53daa4b2066c5361aacb45fa2575885f70c223cda1d11df53d6f)
    FetchContent_MakeAvailable(isaacteleop_cuda_cudart isaacteleop_cuda_nvcc)
    # NVIDIA documents a collapsed SDK prefix for FindCUDAToolkit archive imports.
    # Keep outputs absolute under this build; a consuming project's working
    # directory must not redirect dependency files or SDK discovery.
    set(CUDAToolkit_ROOT "${CMAKE_BINARY_DIR}/_deps/isaacteleop_cuda_sdk")
    file(MAKE_DIRECTORY "${CUDAToolkit_ROOT}")
    file(COPY "${isaacteleop_cuda_cudart_SOURCE_DIR}/" DESTINATION "${CUDAToolkit_ROOT}")
    file(COPY "${isaacteleop_cuda_nvcc_SOURCE_DIR}/" DESTINATION "${CUDAToolkit_ROOT}")
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "OVXR camera quads: build-managed CUDA ${CUDAToolkit_VERSION}")
endif()
