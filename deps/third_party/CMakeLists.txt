# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

include(FetchContent)

# Vendored third-party sources, checked in rather than fetched.
add_subdirectory(openxr_extensions)

# ==============================================================================
# CMake sanitizers
# ==============================================================================
message(STATUS "Fetching sanitizers-cmake from GitHub...")
FetchContent_Declare(
    sanitizers-cmake
    GIT_REPOSITORY https://github.com/arsenm/sanitizers-cmake.git
    GIT_TAG        bcb1fc68616e9645ca5acea2992412606373ab04
)
FetchContent_MakeAvailable(sanitizers-cmake)
list(APPEND CMAKE_MODULE_PATH ${sanitizers-cmake_SOURCE_DIR}/cmake)

# ==============================================================================
# OpenXR SDK
# ==============================================================================
message(STATUS "Fetching OpenXR SDK from GitHub...")
FetchContent_Declare(
    openxr-sdk
    GIT_REPOSITORY https://github.com/KhronosGroup/OpenXR-SDK.git
    GIT_TAG        75c53b6e853dc12c7b3c771edc9c9c841b15faaa  # release-1.1.53
    GIT_SHALLOW    FALSE  # Need full history for this specific commit
)

# Build OpenXR loader as a static library to avoid runtime dependency issues
set(DYNAMIC_LOADER OFF CACHE BOOL "Build OpenXR loader as static library" FORCE)

FetchContent_MakeAvailable(openxr-sdk)
message(STATUS "Building OpenXR SDK from FetchContent (static library)")

# ==============================================================================
# yaml-cpp
# ==============================================================================
# yaml-cpp is used for parsing YAML configuration files
# Build as a static library to avoid runtime dependency issues
message(STATUS "Fetching yaml-cpp from GitHub...")
FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        bbf8bdb087bb3f3621ca0a5ace06047805f4e9f3  # master as of Dec 1, 2025
    GIT_SHALLOW    FALSE  # Need full history for this specific commit
)

# Configure yaml-cpp build options
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Enable testing" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Enable parse tools" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Enable contrib stuff in library" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "Enable generation of install target" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "Build yaml-cpp as shared library" FORCE)

FetchContent_MakeAvailable(yaml-cpp)
message(STATUS "Building yaml-cpp from FetchContent (static library)")

# ==============================================================================
# pybind11
# ==============================================================================
# pybind11 is used for Python bindings (header-only library)
# Only add if Python bindings are enabled
if(BUILD_PYTHON_BINDINGS)
    message(STATUS "Fetching pybind11 from GitHub...")
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        a2e59f0e7065404b44dfe92a28aca47ba1378dc4  # v2.11.0-182
        GIT_SHALLOW    FALSE  # Need full history for this specific commit
    )

    FetchContent_MakeAvailable(pybind11)
    message(STATUS "Using pybind11 from FetchContent")

    # Verify pybind11 targets are available
    if(NOT TARGET pybind11::module)
        message(FATAL_ERROR "pybind11::module target not found after FetchContent")
    endif()
endif()

# ==============================================================================
# FlatBuffers
# ==============================================================================
# FlatBuffers is used for efficient serialization of messages.
FetchContent_Declare(
    flatbuffers
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG v24.3.25
)

# Configure FlatBuffers build options.
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "Enable FlatBuffers tests" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "Enable FlatBuffers install" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "Build flatc compiler" FORCE)

FetchContent_MakeAvailable(flatbuffers)
message(STATUS "FlatBuffers fetched and configured")

# ==============================================================================
# spdlog (C++ logging library)
# ==============================================================================
# Backs isaaccapture::Logger (src/core/log_bridge). Needed by standalone plugin
# executables regardless of BUILD_PYTHON_BINDINGS, so fetched unconditionally
# like FlatBuffers/MCAP rather than gated.
message(STATUS "Fetching spdlog from GitHub...")
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        79524ddd08a4ec981b7fea76afd08ee05f83755d  # v1.17.0
    GIT_SHALLOW    FALSE  # Need full history for this specific commit
)
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "Skip spdlog examples" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "Skip spdlog tests" FORCE)
set(SPDLOG_INSTALL OFF CACHE BOOL "Skip spdlog install target" FORCE)
# DepthAI calls find_package(spdlog CONFIG REQUIRED) itself, which silently
# returns using the targets defined here, so it ends up compiled against *this*
# spdlog. vcpkg's spdlog port is built with SPDLOG_FMT_EXTERNAL=ON and ships no
# bundled fmt, and the bundled headers carry the upstream include guards, so
# mixing the two leaves DepthAI's translation units seeing different spdlog/fmt
# types than ours were compiled with. Pointing at the vcpkg fmt leaves one fmt
# in the build.
#
# Set in both directions, because FORCE writes the cache: setting it only in the
# ON branch left an earlier -DBUILD_PLUGIN_OAK_CAMERA=ON standing in a build
# directory reconfigured with the option off.
if(BUILD_PLUGIN_OAK_CAMERA)
    # Explicit, though spdlog would find_package(fmt) on its own: a missing fmt
    # should fail here, naming this as the reason, not inside spdlog's build.
    find_package(fmt CONFIG REQUIRED)
    set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "Share DepthAI's vcpkg fmt" FORCE)
    message(STATUS "spdlog will use the external (vcpkg) fmt, shared with DepthAI")
else()
    set(SPDLOG_FMT_EXTERNAL OFF CACHE BOOL "Use spdlog's bundled fmt" FORCE)
endif()
FetchContent_MakeAvailable(spdlog)
message(STATUS "spdlog v1.17.0 fetched")

# ==============================================================================
# Catch2 (Testing Framework)
# ==============================================================================
# Catch2 is a modern, C++-native, test framework for unit tests.
# Only fetch if testing is enabled.
if(BUILD_TESTING)
    message(STATUS "Fetching Catch2 from GitHub...")
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.11.0
        GIT_SHALLOW    TRUE
    )

    FetchContent_MakeAvailable(Catch2)
    message(STATUS "Catch2 fetched and configured")

    # Make Catch2's CMake modules available for catch_discover_tests
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)

    include(CTest)
    include(Catch)
endif()

# ==============================================================================
# MCAP (Recording Format Library)
# ==============================================================================
# MCAP is a modular container format for heterogeneous timestamped data.
# It is a header-only library.
message(STATUS "Fetching MCAP from GitHub...")
FetchContent_Declare(
    mcap
    GIT_REPOSITORY https://github.com/foxglove/mcap.git
    GIT_TAG        releases/cpp/v2.1.1
    GIT_SHALLOW    TRUE
)

FetchContent_MakeAvailable(mcap)
message(STATUS "MCAP fetched")

# Create an interface library for MCAP (header-only)
add_library(mcap_interface INTERFACE)
target_include_directories(mcap_interface INTERFACE
    ${mcap_SOURCE_DIR}/cpp/mcap/include
)
# Disable compression to avoid Zstd/LZ4 dependencies.
# Override MCAP_PUBLIC to empty on Windows: upstream visibility.hpp defaults to
# __declspec(dllimport) for TUs without MCAP_IMPLEMENTATION, but we compile MCAP
# into static libraries — dllimport decorations produce unresolvable __imp_ refs.
target_compile_definitions(mcap_interface INTERFACE
    MCAP_COMPRESSION_NO_ZSTD
    MCAP_COMPRESSION_NO_LZ4
    $<$<PLATFORM_ID:Windows>:MCAP_PUBLIC=>
)
add_library(mcap::mcap ALIAS mcap_interface)
message(STATUS "MCAP interface library configured")

# ==============================================================================
# GLM (Vulkan/GLSL math)
# ==============================================================================
# Header-only math library used by the viz module for matrix / vector / quaternion
# math (world-lock, head-lock, lazy-lock layer placement; view & projection
# composition; quaternion slerp). Conventions match GLSL/Vulkan (column-major,
# default precision float). Memory layout is POD-compatible with raw float[]:
# `glm::value_ptr(mat)` returns `float*` for direct upload to Vulkan/CUDA buffers.
if(BUILD_VIZ)
    message(STATUS "Fetching glm from GitHub...")
    FetchContent_Declare(
        glm
        GIT_REPOSITORY https://github.com/g-truc/glm.git
        GIT_TAG        1.0.1
        GIT_SHALLOW    TRUE
    )
    set(GLM_BUILD_LIBRARY OFF CACHE BOOL "Build glm as header-only" FORCE)
    set(GLM_BUILD_TESTS OFF CACHE BOOL "Skip glm test suite" FORCE)
    FetchContent_MakeAvailable(glm)
    message(STATUS "glm 1.0.1 fetched (header-only)")
endif()

# ==============================================================================
# GLFW (window + Vulkan surface for kWindow)
# ==============================================================================
# Owns GLFWwindow + VkSurfaceKHR for VizSession's kWindow display backend.
# Static build to avoid runtime .so dependency.
if(BUILD_VIZ)
    message(STATUS "Fetching GLFW from GitHub...")
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
        GIT_SHALLOW    TRUE
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "Skip GLFW docs" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "Skip GLFW tests" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Skip GLFW examples" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "Skip GLFW install target" FORCE)
    # Build with both X11 and Wayland (GLFW 3.4 Linux defaults).
    # Requires libxrandr-dev / libxinerama-dev / libxcursor-dev /
    # libxi-dev / libxext-dev / libxkbcommon-dev plus libwayland-dev /
    # wayland-scanner. CI installs them in build-ubuntu.yml. Override
    # with -DGLFW_BUILD_WAYLAND=OFF on hosts without Wayland tooling.
    FetchContent_MakeAvailable(glfw)
    message(STATUS "GLFW 3.4 fetched")
endif()

# ==============================================================================
# MuJoCo (robot twin scene + renderer)
# ==============================================================================
# Built from upstream sources unmodified and shipped under a private name, so the robot
# twin's MuJoCo is an implementation detail nothing else in the process can see.
# Mujoco.cmake beside this file is that contract and the function consumers reach it
# through; src/viz/robot_twin is the only consumer.
#
# Upstream builds one library -- engine, renderer, MJCF parser -- with no renderer-only
# target, so this is the whole of it even though the twin calls no dynamics.
# src/viz/robot_twin is the only consumer and it is a pybind11 extension, so a build
# without the bindings has nothing to link this to.
if(BUILD_VIZ AND BUILD_PYTHON_BINDINGS)
    message(STATUS "Fetching MuJoCo from GitHub...")
    FetchContent_Declare(
        mujoco
        GIT_REPOSITORY https://github.com/google-deepmind/mujoco.git
        GIT_TAG        3.11.0
        GIT_SHALLOW    TRUE
    )

    # Before MakeAvailable, which is what runs MuJoCo's own CMakeLists -- so these cannot
    # move into Mujoco.cmake below. Plain variables, not the `CACHE ... FORCE` the entries
    # above use: MuJoCo sets CMAKE_POLICY_DEFAULT_CMP0077 NEW before its own project(), so
    # a normal variable wins over its option() and leaves no knob in the cache. Turning the
    # tests off is what keeps abseil, googletest and benchmark out of _deps.
    set(MUJOCO_BUILD_EXAMPLES OFF)
    set(MUJOCO_BUILD_SIMULATE OFF)
    set(MUJOCO_BUILD_TESTS OFF)
    set(MUJOCO_TEST_PYTHON_UTIL OFF)

    FetchContent_MakeAvailable(mujoco)
    message(STATUS "MuJoCo 3.11.0 fetched")

    include("${CMAKE_CURRENT_LIST_DIR}/Mujoco.cmake")
endif()

# ==============================================================================
# SPDX license list data
# ==============================================================================
# Reference license texts. The SBOM collector names a dependency's license by
# matching the text that dependency ships against these; nothing from this
# checkout is redistributed, and it builds nothing. Pinned by commit, so no tag
# upstream can move underneath a build.
find_package(Git REQUIRED)

set(SPDX_LICENSE_DATA_COMMIT "31ba1a50e5397e00a304dbadc76531740e89ee48")  # v3.29.0
set(SPDX_LICENSE_DATA_SOURCE_DIR "${FETCHCONTENT_BASE_DIR}/license-list-data-src")

message(STATUS "Fetching SPDX license list data from GitHub...")
FetchContent_Declare(
    license_list_data
    SOURCE_DIR "${SPDX_LICENSE_DATA_SOURCE_DIR}"
    DOWNLOAD_COMMAND
        "${CMAKE_COMMAND}"
        "-DGIT_EXECUTABLE=${GIT_EXECUTABLE}"
        "-DSPDX_REPOSITORY=https://github.com/spdx/license-list-data.git"
        "-DSPDX_COMMIT=${SPDX_LICENSE_DATA_COMMIT}"
        "-DSPDX_SOURCE_DIR=${SPDX_LICENSE_DATA_SOURCE_DIR}"
        -P "${CMAKE_CURRENT_LIST_DIR}/fetch_license_list_data.cmake"
)
FetchContent_MakeAvailable(license_list_data)

# What the SBOM collector reads, and what the sbom tests run against.
set(SPDX_LICENSE_DATA_JSON_DIR "${SPDX_LICENSE_DATA_SOURCE_DIR}/json"
    CACHE INTERNAL "SPDX reference license texts for the SBOM collector")
if(NOT EXISTS "${SPDX_LICENSE_DATA_JSON_DIR}/licenses.json")
    message(FATAL_ERROR "SPDX license list data is missing json/licenses.json after fetch")
endif()
