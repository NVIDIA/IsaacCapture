# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# SENSING Camera Plugin (Jetson SIPL capture + V4L2 H.264 encode)
# ==============================================================================
# Jetson-only: SIPL and the V4L2 M2M encoder exist only on L4T.
#
# Four leaf targets, one per directory (cmake/cmake-structure.md rule 1):
#   cuda_ipc/  sensing_cuda_ipc  — CUDA-only IPC publisher + wire format
#   core/      sensing_core      — SIPL capture, encoder, sinks
#   app/       camera_plugin_sensing
#   tools/     sensing_ipc_testsrc — links cuda_ipc only, no SIPL, no encoder
#
# The Jetson SDK checks below gate the whole plugin, tools/ included.
#
# Prerequisites:
#   src/plugins/sensing/setup_container.sh   # fetches the Multimedia API,
#                                            # locates jetson_sipl_api
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

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_tegra_lib_dirs "/usr/lib/aarch64-linux-gnu/nvidia" "/usr/lib/aarch64-linux-gnu/tegra")
else()
    set(_tegra_lib_dirs "/usr/lib/x86_64-linux-gnu/nvidia")
endif()

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
        "  src/plugins/sensing/setup_container.sh\n"
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
        "  src/plugins/sensing/setup_container.sh\n"
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
# Targets
# ==============================================================================
message(STATUS "Building SENSING camera plugin")

add_subdirectory(cuda_ipc)
add_subdirectory(core)
add_subdirectory(app)
add_subdirectory(tools)

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/plugin.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/README.md"
    DESTINATION plugins/sensing_camera
)
