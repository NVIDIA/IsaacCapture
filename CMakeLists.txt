# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.24...3.25)

# Workaround for older CMake policy compatibility in dependencies
set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum CMake policy version")

include(cmake/IsaacTeleopVersion.cmake)
isaac_teleop_read_version("${CMAKE_CURRENT_SOURCE_DIR}/VERSION" ISAAC_TELEOP_VERSION ISAAC_TELEOP_PYPROJECT_VERSION)

# ==============================================================================
# Install Prefix Setup
# ==============================================================================
if(NOT DEFINED CMAKE_INSTALL_PREFIX OR CMAKE_INSTALL_PREFIX STREQUAL "" OR CMAKE_INSTALL_PREFIX MATCHES "^/usr/local")
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/install" CACHE PATH "Install prefix" FORCE)
endif()

# ==============================================================================
# DepthAI vcpkg manifest (OAK camera plugin)
# ==============================================================================
# The OAK camera plugin's vcpkg dependencies come from DepthAI's own vcpkg.json,
# auto-downloaded for the pinned tag, rather than a duplicate manifest tracked
# in this repo. This MUST run before project() so vcpkg picks up the manifest
# when the toolchain loads. Gated on the -D option being passed (it lives in the
# cache before project(); the option() default below applies only when unset).
if(BUILD_PLUGIN_OAK_CAMERA)
    include(cmake/DepthAIVcpkgManifest.cmake)
    isaac_teleop_setup_depthai_vcpkg_manifest()
endif()

project(IsaacTeleop VERSION ${ISAAC_TELEOP_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==============================================================================
# Python Version Configuration (SINGLE SOURCE OF TRUTH)
# ==============================================================================
# All Python-related configurations should reference this variable.
# To change the Python version, only modify this line:
set(ISAAC_TELEOP_PYTHON_VERSION "3.11" CACHE STRING "Python version for Isaac Teleop")
set(ISAAC_TELEOP_PYTHON_VERSION_MIN "3.11")
set(ISAAC_TELEOP_PYTHON_VERSION_MAX_EXCLUSIVE "3.14")
message(STATUS "Configuring for Python ${ISAAC_TELEOP_PYTHON_VERSION}")
# ==============================================================================

# Configure Python via uv BEFORE any dependencies are added
# This ensures all subdirectories (including openxr-sdk) use the managed Python
include(cmake/SetupPython.cmake)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(BUILD_PLUGINS "Build plugins" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings" ON)
option(BUILD_PLUGIN_OAK_CAMERA "Build OAK camera plugin (requires vcpkg for DepthAI v3.x)" OFF)
option(BUILD_PLUGIN_NOITOM_MOCAP "Build Noitom mocap plugin (downloads MocapApi SDK)" OFF)
option(BUILD_PLUGIN_OGLO "Build OGLO tactile glove plugin (BLE, Linux only; fetches SimpleBLE + nlohmann/json)" OFF)
option(BUILD_PLUGIN_WUJI_GLOVE "Build Wuji glove plugin (requires the wuji_sdk C SDK)" OFF)
option(BUILD_PLUGIN_IO_EXO_GLOVE_ROS2 "Build exoskeleton glove ROS 2 (Humble) bridge plugin (Linux only; requires a sourced ROS 2 environment)" OFF)
option(BUILD_EXAMPLES "Build examples" ON)
option(BUILD_EXAMPLE_TELEOP_ROS2 "Build only the teleop_ros2 ROS 2 reference integration (e.g. for Docker)" OFF)
option(BUILD_TESTING "Build unit tests" ON)
# Televiz (src/viz) needs Vulkan, the CUDA Toolkit, and glslangValidator.
# Only Vulkan and CUDA gate the default -- without them the machine cannot build
# viz at all, so a core-only source build still configures. glslangValidator is
# an apt package, not a GPU stack, so CheckBuildDeps fails on a missing one
# rather than dropping the module. Override with -DBUILD_VIZ=ON/OFF.
find_package(Vulkan QUIET)
find_package(CUDAToolkit QUIET)
if(Vulkan_FOUND AND CUDAToolkit_FOUND)
    set(_viz_default ON)
else()
    set(_viz_default OFF)
endif()
option(BUILD_VIZ "Build Televiz visualization module (auto-ON when Vulkan + CUDA are found)" ${_viz_default})
if(BUILD_VIZ AND NOT (Vulkan_FOUND AND CUDAToolkit_FOUND))
    message(WARNING "BUILD_VIZ=ON but Vulkan / the CUDA Toolkit were not detected; the viz build "
                    "may fail. Install the deps or pass -DBUILD_VIZ=OFF.")
endif()
message(STATUS "BUILD_VIZ: ${BUILD_VIZ} (Vulkan=${Vulkan_FOUND} CUDAToolkit=${CUDAToolkit_FOUND})")
option(ENABLE_EXPERIMENTAL_WINDOWS_BUILD "Enable experimental Windows build support (source code only)" OFF)

# Note: This is used to check if the CloudXR bundle is available to be bundled with Isaac Teleop.
# When enabled, build will fail if the CloudXR runtime SDK is not available.
# TODO: Make this default to ON for Linux builds when the CloudXR runtime SDK is publicly available.
option(ENABLE_CLOUDXR_BUNDLE_CHECK "Enable CloudXR bundle check" OFF)

# When ON, also package CloudXR Experimental Runtime SDK as isaaccapture.cloudxr_exp.
option(ENABLE_CLOUDXR_EXP_BUNDLE "Also package CloudXR Experimental Runtime SDK as isaaccapture.cloudxr_exp" ON)

## Enforce clang-format on Linux builds by default
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_clang_default ON)
else()
    set(_clang_default OFF)
endif()
option(ENABLE_CLANG_FORMAT_CHECK "Fail build if clang-format changes would be applied (Linux default ON)" ${_clang_default})

# Windows support warning
if(WIN32)
    if(NOT ENABLE_EXPERIMENTAL_WINDOWS_BUILD)
        message(FATAL_ERROR
            "================================================================================\n"
            "Windows is not officially supported!\n"
            "If you want to build anyway for development/testing purposes, add:\n"
            "  -DENABLE_EXPERIMENTAL_WINDOWS_BUILD=ON\n"
            "Use at your own risk!\n"
            "================================================================================\n"
        )
    else()
        # Create a function to print the warning cleanly
        function(print_windows_warning)
            message(STATUS "")
            message(STATUS "================================================================================")
            message(STATUS "⚠️  EXPERIMENTAL WINDOWS BUILD ENABLED - USE AT YOUR OWN RISK ⚠️")
            message(STATUS "")
            message(STATUS "You have enabled experimental Windows build support.")
            message(STATUS "The source code will compile but there is NO guarantee of any functionality.")
            message(STATUS "This is for development and testing purposes only!")
            message(STATUS "================================================================================")
        endfunction()

        # Defer the warning to the end of configuration so it's highly visible
        cmake_language(DEFER CALL print_windows_warning)
    endif()
endif()

# Fail on missing build tools now, while the feature set is known and before
# any FetchContent, rather than 60 s deep in a fetched dependency.
include(cmake/CheckBuildDeps.cmake)
isaac_teleop_check_build_deps()

# Project warning set. This only defines the helpers -- src/CMakeLists.txt and
# examples/CMakeLists.txt each enable the set for their own tree, so the flags are
# scoped by directory rather than by where this include sits. Nothing here applies
# compile options, and deps/ below is unaffected.
include(cmake/CompilerWarnings.cmake)

# Build dependencies (OpenXR SDK, etc.)
add_subdirectory(deps)

# Enable CTest at top level so tests from subdirectories are discoverable
if(BUILD_TESTING)
    # Make sure to call this after `deps` is added so that Catch2 is available
    enable_testing()
endif()

# Build interfaces (headers)
add_subdirectory(deps/cloudxr/openxr_extensions)

# Build the first-party tree: core modules, the Python package, Televiz, plugins.
add_subdirectory(src)

# Build examples if requested (examples/CMakeLists.txt honours both options).
if(BUILD_EXAMPLES OR BUILD_EXAMPLE_TELEOP_ROS2)
    add_subdirectory(examples)
endif()

# Tests (pytest + Catch2/CTest). After production libs so link targets exist.
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# Formatting enforcement (runs on Linux by default)
include(cmake/ClangFormat.cmake)

# ==============================================================================
# scikit-build-core: install every built executable into the venv's bin/
# ==============================================================================
# Under pip (SKBUILD), put all executables built from this repo -- examples, C++
# test binaries, plugin tools, etc. -- on the venv PATH via the wheel's scripts
# scheme. We walk the directory tree from the project root but skip deps/ (and any
# FetchContent _deps/ subtree) so third-party executables (e.g. flatc) are never
# swept in. Shared libraries (Python extension modules, plugin .so) are not
# executables and are handled elsewhere / not installed here; a plugin that skips a
# missing SDK becomes a UTILITY target and is naturally excluded. Tagged with the
# isaacteleop_binaries component, selected via install.components in pyproject.toml.
#
# A target that sets ISAACTELEOP_NEEDS_SOURCE_TREE is skipped: it reads repository
# files at run time, which the wheel does not carry, so installing it would ship a
# binary holding paths into the machine that built it.
if(SKBUILD)
    function(isaac_teleop_install_executables _dir)
        # Skip third-party dependency trees (deps/ and FetchContent's _deps/).
        if(_dir STREQUAL "${CMAKE_SOURCE_DIR}/deps" OR _dir MATCHES "/_deps/")
            return()
        endif()
        get_property(_subdirs DIRECTORY "${_dir}" PROPERTY SUBDIRECTORIES)
        foreach(_subdir IN LISTS _subdirs)
            isaac_teleop_install_executables("${_subdir}")
        endforeach()
        get_property(_targets DIRECTORY "${_dir}" PROPERTY BUILDSYSTEM_TARGETS)
        foreach(_target IN LISTS _targets)
            get_target_property(_type "${_target}" TYPE)
            get_target_property(_imported "${_target}" IMPORTED)
            get_target_property(_needs_source_tree "${_target}" ISAACTELEOP_NEEDS_SOURCE_TREE)
            if(_type STREQUAL "EXECUTABLE" AND NOT _imported AND NOT _needs_source_tree)
                install(TARGETS "${_target}"
                    RUNTIME DESTINATION "${SKBUILD_SCRIPTS_DIR}"
                    COMPONENT isaacteleop_binaries)
            endif()
        endforeach()
    endfunction()

    # Walk from the project root (deps/ skipped) so every executable built from
    # this repo -- across src/ and examples/ -- is installed.
    isaac_teleop_install_executables("${CMAKE_SOURCE_DIR}")
endif()
