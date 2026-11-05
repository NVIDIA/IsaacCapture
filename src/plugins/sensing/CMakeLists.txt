# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# SENSING Camera Plugin: Jetson SIPL capture, hardware H.264 encode, LiveKit publication
# ==============================================================================
# Two leaf targets besides the LiveKit wrapper, one per directory (cmake/cmake-structure.md rule 1):
#   livekit/  sensing_livekit       -- the LiveKit SDK behind a small interface
#   core/     sensing_core          -- SIPL capture, encoder, conversion, track output
#   app/      camera_plugin_sensing
#
# Prerequisites:
#   src/plugins/sensing/scripts/setup.sh    # fetches the Multimedia API, locates jetson_sipl_api
#
# Configure with:
#   cmake -B build -DBUILD_PLUGIN_SENSING=ON
# ==============================================================================

enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Orin is 87, Thor is 101; the rest cover desktop dev boxes.
    set(CMAKE_CUDA_ARCHITECTURES "80;86;87;89;90")
endif()

find_package(CUDAToolkit REQUIRED)

set(_tegra_lib_dirs "/usr/lib/aarch64-linux-gnu/nvidia" "/usr/lib/aarch64-linux-gnu/tegra")

# ==============================================================================
# SIPL
# ==============================================================================
# Headers ship in Jetson_SIPL_API_R<rel>_aarch64.tbz2, published alongside the
# L4T release. Not in any apt pool, so setup_container.sh fetches it by URL.
if(NOT SIPL_API_DIR)
    find_path(SIPL_API_DIR sipl/include/NvSIPLCamera.hpp
        PATHS "/usr/src/jetson_sipl_api"
              "/opt/nvidia/jetson_sipl_api"
              "$ENV{SIPL_API_DIR}"
    )
endif()

if(NOT SIPL_API_DIR)
    message(FATAL_ERROR
        "================================================================================\n"
        "SENSING camera plugin requires the Jetson SIPL API headers.\n"
        "\n"
        "Run the container setup, which fetches them:\n"
        "  src/plugins/sensing/scripts/setup.sh\n"
        "\n"
        "Or point CMake at an existing copy:\n"
        "  cmake -B build -DSIPL_API_DIR=/path/to/jetson_sipl_api\n"
        "\n"
        "Or disable this plugin:\n"
        "  cmake -B build -DBUILD_PLUGIN_SENSING=OFF\n"
        "================================================================================\n"
    )
endif()

set(SIPL_INCLUDE_DIRS
    "${SIPL_API_DIR}/sipl/include"
    "${SIPL_API_DIR}/sipl/include/query/include"
    "${SIPL_API_DIR}/sipl/include/nvsci"
)
message(STATUS "SENSING plugin: SIPL=${SIPL_API_DIR}")

# L4T ships no .so dev symlinks for these, hence the explicit sonames.
foreach(_lib nvsipl nvsipl_query nvscibuf nvscisync nvbufsurface_nvsci)
    string(TOUPPER "${_lib}" _upper)
    find_library(${_upper}_LIBRARY
        NAMES ${_lib} lib${_lib}.so.1 lib${_lib}.so.1.0.0
        PATHS ${_tegra_lib_dirs}
    )
    if(NOT ${_upper}_LIBRARY)
        message(FATAL_ERROR
            "lib${_lib} not found. It is supplied by the L4T BSP and, in a container, by the "
            "NVIDIA container runtime. Run src/plugins/sensing/verify.sh to see what is missing.")
    endif()
    list(APPEND SIPL_LIBRARIES ${${_upper}_LIBRARY})
endforeach()

# EGL stays on the capture path: NvBufSurfaceMapEglImage is how a SIPL
# NvSciBufObj becomes a CUeglFrame. Interop path B in the design notes would
# remove this; path A does not.
find_path(EGL_INCLUDE_DIRS EGL/egl.h)
find_library(EGL_LIBRARIES NAMES EGL)
if(NOT EGL_INCLUDE_DIRS OR NOT EGL_LIBRARIES)
    message(FATAL_ERROR "SENSING camera plugin requires EGL. Install libegl1-mesa-dev.")
endif()

# ==============================================================================
# Jetson V4L2 encoder
# ==============================================================================
# Jetson has no libnvidia-encode (the Video Codec SDK is dGPU-only), so H.264
# comes from the V4L2 M2M engine via NvVideoEncoder in the Multimedia API.
if(NOT JETSON_MMAPI_DIR)
    find_path(JETSON_MMAPI_DIR include/NvVideoEncoder.h
        PATHS "/usr/src/jetson_multimedia_api"
              "$ENV{HOME}/Sensing/jetson_multimedia_api"
    )
endif()
if(NOT JETSON_MMAPI_DIR)
    message(FATAL_ERROR
        "================================================================================\n"
        "SENSING camera plugin requires the Jetson Multimedia API encoder sources.\n"
        "\n"
        "setup_container.sh fetches the version matching /etc/nv_tegra_release:\n"
        "  src/plugins/sensing/scripts/setup.sh\n"
        "\n"
        "Or point CMake at an existing tree:\n"
        "  cmake -B build -DJETSON_MMAPI_DIR=/path/to/jetson_multimedia_api\n"
        "================================================================================\n"
    )
endif()
message(STATUS "SENSING plugin: Jetson Multimedia API=${JETSON_MMAPI_DIR}")

set(_mmapi_classes "${JETSON_MMAPI_DIR}/samples/common/classes")
set(JETSON_MMAPI_SOURCES
    "${_mmapi_classes}/NvVideoEncoder.cpp"
    "${_mmapi_classes}/NvV4l2Element.cpp"
    "${_mmapi_classes}/NvV4l2ElementPlane.cpp"
    "${_mmapi_classes}/NvElement.cpp"
    "${_mmapi_classes}/NvElementProfiler.cpp"
    "${_mmapi_classes}/NvBuffer.cpp"
    "${_mmapi_classes}/NvLogging.cpp"
)

find_library(NVBUFSURFACE_LIBRARY NAMES nvbufsurface PATHS ${_tegra_lib_dirs})
if(NOT NVBUFSURFACE_LIBRARY)
    message(FATAL_ERROR "libnvbufsurface not found. Install nvidia-l4t-multimedia.")
endif()

# NVIDIA's patched libv4l2 carries the Tegra M2M encoder ioctls, so search the
# tegra directories before the generic one. L4T ships no .so dev symlink, hence
# the explicit soname in NAMES.
find_library(V4L2_LIBRARY
    NAMES v4l2 libv4l2.so.0
    PATHS ${_tegra_lib_dirs}
    NO_DEFAULT_PATH
)
if(NOT V4L2_LIBRARY)
    find_library(V4L2_LIBRARY NAMES v4l2 libv4l2.so.0)
endif()
if(NOT V4L2_LIBRARY)
    message(FATAL_ERROR "libv4l2 not found. Install nvidia-l4t-multimedia (or libv4l-dev).")
endif()
message(STATUS "SENSING plugin: V4L2=${V4L2_LIBRARY}")

# No libnvjpeg here on purpose: on R39 libnvbufsurface has no libjpeg
# dependency at all (readelf -d lists only EGL, cuda, dl, pthread, rt, c), and
# the tegra lib dir ships no libnvjpeg to link against.

# ==============================================================================
# LiveKit
# ==============================================================================
# Each selected sensor is published as a LiveKit camera track. The SDK is a prebuilt package (headers and shared libraries), pinned to a
# release and checked against its hash; point LIVEKIT_SDK_DIR at an unpacked copy to use another.
if(NOT LIVEKIT_SDK_DIR)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(_livekit_arch "arm64")
        set(_livekit_hash "e3a97f71943c0bf9fa4ebd70927ca2906b31dcd6c14b42312b1225b4b868e4ad")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(_livekit_arch "x64")
        set(_livekit_hash "5e0feb2e519c6ff131dd2d1c45149115057ae4e21a324fdb249e1789d5614e7b")
    else()
        message(FATAL_ERROR "No prebuilt LiveKit SDK for ${CMAKE_SYSTEM_PROCESSOR}; set LIVEKIT_SDK_DIR.")
    endif()
    include(FetchContent)
    FetchContent_Declare(
        livekit_sdk
        URL https://github.com/livekit/client-sdk-cpp/releases/download/v1.12.4/livekit-sdk-ubuntu-24.04-${_livekit_arch}-1.12.4.tar.gz
        URL_HASH SHA256=${_livekit_hash}
    )
    FetchContent_MakeAvailable(livekit_sdk)
    set(LIVEKIT_SDK_DIR "${livekit_sdk_SOURCE_DIR}")
endif()
# The package ships no licence files: install the SDK's own, pinned to the same release.
foreach(_notice LICENSE NOTICE)
    if(_notice STREQUAL "LICENSE")
        set(_notice_hash "09e8a9bcec8067104652c168685ab0931e7868f9c8284b66f5ae6edae5f1130b")
    else()
        set(_notice_hash "6d0f7c5e0deb284362296c5da110dd353831da2e67ca77cb5abc051b951b0fac")
    endif()
    FetchContent_Declare(
        livekit_sdk_${_notice}
        URL https://raw.githubusercontent.com/livekit/client-sdk-cpp/v1.12.4/${_notice}
        URL_HASH SHA256=${_notice_hash}
        DOWNLOAD_NO_EXTRACT TRUE
    )
    FetchContent_MakeAvailable(livekit_sdk_${_notice})
    set(SENSING_LIVEKIT_${_notice}_FILE "${livekit_sdk_${_notice}_SOURCE_DIR}/${_notice}" CACHE INTERNAL "")
endforeach()
find_package(LiveKit CONFIG REQUIRED PATHS "${LIVEKIT_SDK_DIR}" NO_DEFAULT_PATH)
message(STATUS "SENSING plugin: LiveKit SDK=${LIVEKIT_SDK_DIR}")
# Cache, so tests/ (a sibling directory) can find the same SDK.
set(SENSING_LIVEKIT_SDK_DIR "${LIVEKIT_SDK_DIR}" CACHE INTERNAL "")


# ==============================================================================
# Targets
# ==============================================================================
message(STATUS "Building SENSING camera plugin")

add_subdirectory(livekit)
add_subdirectory(core)
add_subdirectory(app)

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    DESTINATION plugins/sensing_camera
)
