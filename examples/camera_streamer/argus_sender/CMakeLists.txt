# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

project(argus_sender LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
include(CTest)

option(DISABLE_MULTIPROCESS
    "Link the single-process Argus implementation instead of nvargus-daemon's socket client"
    OFF)

option(WITH_SIPL "Build the SIPL capture backend (needs the SIPL API and NvSci libraries)" OFF)
set(SIPL_ROOT "/usr/src/jetson_sipl_api/sipl" CACHE PATH "SIPL API root (WITH_SIPL=ON)")

set(MMAPI_ROOT "/usr/src/jetson_multimedia_api" CACHE PATH
    "Jetson Multimedia API root")
set(MMAPI_CLASS_DIR "${MMAPI_ROOT}/samples/common/classes" CACHE PATH
    "Jetson Multimedia API common/classes directory")

set(NVIDIA_LIBRARY_DIRS
    /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia
    /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/tegra
    /usr/lib/aarch64-linux-gnu/nvidia
    /usr/lib/aarch64-linux-gnu/tegra
    /opt/nvidia/l4t-gpu-libs/openrm
    /opt/nvidia/l4t-gpu-libs/nvgpu
)

find_path(ARGUS_INCLUDE_DIR Argus/Argus.h
          HINTS "${MMAPI_ROOT}/argus/include" "${CMAKE_CURRENT_SOURCE_DIR}/include")
find_library(ARGUS_SOCKETCLIENT_LIBRARY NAMES nvargus_socketclient
             HINTS ${NVIDIA_LIBRARY_DIRS})
find_library(ARGUS_LIBRARY NAMES nvargus
             HINTS ${NVIDIA_LIBRARY_DIRS})

find_path(MMAPI_INCLUDE_DIR NvVideoEncoder.h
          HINTS "${MMAPI_ROOT}/include")
find_path(CUDA_INCLUDE_DIR cuda.h
          HINTS /usr/local/cuda/include /usr/include)
find_path(DRM_INCLUDE_DIR drm.h
          HINTS /usr/include/libdrm)

find_library(EGL_LIBRARY NAMES EGL)
find_library(NVV4L2_LIBRARY NAMES nvv4l2
             HINTS ${NVIDIA_LIBRARY_DIRS})
find_library(NVBUFSURFACE_LIBRARY NAMES nvbufsurface
             HINTS ${NVIDIA_LIBRARY_DIRS})
find_library(CUDA_DRIVER_LIBRARY NAMES cuda
             HINTS ${NVIDIA_LIBRARY_DIRS} /usr/lib/aarch64-linux-gnu)

find_package(Threads REQUIRED)

include(FetchContent)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    yaml_cpp
    URL
        https://github.com/jbeder/yaml-cpp/releases/download/yaml-cpp-0.9.0/yaml-cpp-yaml-cpp-0.9.0.tar.gz
    URL_HASH
        SHA256=298593d9c440fd9034b8b193d96318b76d49bc97c6ceadb7b0836edf0b6d7539
)
FetchContent_MakeAvailable(yaml_cpp)

set(REQUIRED_PATHS
    ARGUS_INCLUDE_DIR
    MMAPI_INCLUDE_DIR
    CUDA_INCLUDE_DIR
    DRM_INCLUDE_DIR
    MMAPI_CLASS_DIR
)
foreach(path_var IN LISTS REQUIRED_PATHS)
    if(NOT ${path_var})
        message(FATAL_ERROR "${path_var} was not found")
    endif()
endforeach()

set(REQUIRED_LIBS
    EGL_LIBRARY
    NVV4L2_LIBRARY
    NVBUFSURFACE_LIBRARY
    CUDA_DRIVER_LIBRARY
)
foreach(lib_var IN LISTS REQUIRED_LIBS)
    if(NOT ${lib_var})
        message(FATAL_ERROR "${lib_var} was not found")
    endif()
endforeach()

if(DISABLE_MULTIPROCESS)
    if(NOT ARGUS_LIBRARY)
        message(FATAL_ERROR
            "DISABLE_MULTIPROCESS=ON requires the single-process nvargus library")
    endif()
    set(ARGUS_LINK_LIBRARY "${ARGUS_LIBRARY}")
    set(ARGUS_LINK_MODE "single-process")
else()
    if(NOT ARGUS_SOCKETCLIENT_LIBRARY)
        message(FATAL_ERROR
            "The multiprocess build requires the nvargus_socketclient library")
    endif()
    set(ARGUS_LINK_LIBRARY "${ARGUS_SOCKETCLIENT_LIBRARY}")
    set(ARGUS_LINK_MODE "multiprocess")
endif()

set(MMAPI_CLASS_SOURCES
    ${MMAPI_CLASS_DIR}/NvBuffer.cpp
    ${MMAPI_CLASS_DIR}/NvElement.cpp
    ${MMAPI_CLASS_DIR}/NvElementProfiler.cpp
    ${MMAPI_CLASS_DIR}/NvLogging.cpp
    ${MMAPI_CLASS_DIR}/NvV4l2Element.cpp
    ${MMAPI_CLASS_DIR}/NvV4l2ElementPlane.cpp
    ${MMAPI_CLASS_DIR}/NvVideoEncoder.cpp
)

foreach(source_file IN LISTS MMAPI_CLASS_SOURCES)
    if(NOT EXISTS "${source_file}")
        message(FATAL_ERROR "Required Multimedia API source not found: ${source_file}")
    endif()
endforeach()

add_executable(argus_sender
    src/main.cpp
    src/Config.cpp
    src/Avs1Wire.cpp
    src/OutputGeometry.cpp
    src/MixedRateScheduler.cpp
    src/CapturePriorityPolicy.cpp
    src/CaptureDomainPlan.cpp
    src/ArgusFrameAdapter.cpp
    src/CaptureBackend.cpp
    src/CaptureManager.cpp
    src/FrameBundleAssembler.cpp
    src/StreamProcessor.cpp
    src/CaptureCoordinator.cpp
    src/SenderSession.cpp
    src/Runtime.cpp
    src/CameraCatalog.cpp
    src/FrameCopyWorker.cpp
    src/NvRemap.cpp
    src/FrameAlignmentPolicy.cpp
    src/StereoAlignmentPolicy.cpp
    src/TcpTransport.cpp
    src/ArgusBufferBinding.cpp
    src/HevcEncoder.cpp
    src/StereoTiming.cpp
    src/EGLGlobal.cpp
    src/NvBufSurfaceLite.cpp
    ${MMAPI_CLASS_SOURCES}
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(argus_sender PRIVATE -Wall -Wextra)
    set_source_files_properties(${MMAPI_CLASS_SOURCES}
        PROPERTIES COMPILE_OPTIONS -w)
endif()

target_compile_definitions(argus_sender PRIVATE NVMMAPI_SUPPORTED)

target_include_directories(argus_sender PRIVATE
    src
    ${ARGUS_INCLUDE_DIR}
    ${MMAPI_INCLUDE_DIR}
    ${MMAPI_CLASS_DIR}
    ${CUDA_INCLUDE_DIR}
    ${DRM_INCLUDE_DIR}
)

target_link_libraries(argus_sender PRIVATE
    ${ARGUS_LINK_LIBRARY}
    ${EGL_LIBRARY}
    ${NVV4L2_LIBRARY}
    ${NVBUFSURFACE_LIBRARY}
    ${CUDA_DRIVER_LIBRARY}
    yaml-cpp::yaml-cpp
    Threads::Threads
)

if(WITH_SIPL)
    find_path(SIPL_INCLUDE_DIR NvSIPLCamera.hpp HINTS "${SIPL_ROOT}/include")
    find_path(SIPL_QUERY_INCLUDE_DIR NvSIPLCameraQuery.hpp HINTS "${SIPL_ROOT}/include/query/include")
    find_path(NVSCI_INCLUDE_DIR nvscibuf.h HINTS "${SIPL_ROOT}/include/nvsci")
    find_library(NVSIPL_LIBRARY NAMES nvsipl HINTS ${NVIDIA_LIBRARY_DIRS})
    find_library(NVSIPL_QUERY_LIBRARY NAMES nvsipl_query HINTS ${NVIDIA_LIBRARY_DIRS})
    find_library(NVSCIBUF_LIBRARY NAMES nvscibuf HINTS ${NVIDIA_LIBRARY_DIRS})
    find_library(NVSCISYNC_LIBRARY NAMES nvscisync HINTS ${NVIDIA_LIBRARY_DIRS})
    find_library(NVBUFSURFACE_NVSCI_LIBRARY NAMES nvbufsurface_nvsci HINTS ${NVIDIA_LIBRARY_DIRS})
    foreach(sipl_var SIPL_INCLUDE_DIR SIPL_QUERY_INCLUDE_DIR NVSCI_INCLUDE_DIR NVSIPL_LIBRARY NVSIPL_QUERY_LIBRARY
                     NVSCIBUF_LIBRARY NVSCISYNC_LIBRARY NVBUFSURFACE_NVSCI_LIBRARY)
        if(NOT ${sipl_var})
            message(FATAL_ERROR "WITH_SIPL=ON but ${sipl_var} was not found")
        endif()
    endforeach()

    # The SIPL headers need C++17; the rest of the sender stays C++14.
    add_library(sipl_capture STATIC src/SiplCapture.cpp)
    set_target_properties(sipl_capture PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    target_compile_options(sipl_capture PRIVATE -Wall -Wextra)
    target_include_directories(sipl_capture PRIVATE
        src
        ${SIPL_INCLUDE_DIR}
        ${SIPL_QUERY_INCLUDE_DIR}
        ${NVSCI_INCLUDE_DIR}
        ${MMAPI_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIR}
    )
    target_link_libraries(sipl_capture PUBLIC
        ${NVSIPL_LIBRARY}
        ${NVSIPL_QUERY_LIBRARY}
        ${NVSCIBUF_LIBRARY}
        ${NVSCISYNC_LIBRARY}
        ${NVBUFSURFACE_LIBRARY}
        ${NVBUFSURFACE_NVSCI_LIBRARY}
        Threads::Threads
    )
    target_sources(argus_sender PRIVATE src/SiplSender.cpp)
    target_compile_definitions(argus_sender PRIVATE SENDER_WITH_SIPL)
    target_link_libraries(argus_sender PRIVATE sipl_capture)
endif()

if(BUILD_TESTING)
    include(cmake/Tests.cmake)
endif()

install(TARGETS argus_sender DESTINATION bin)

message(STATUS "Argus include: ${ARGUS_INCLUDE_DIR}")
message(STATUS "Argus library (${ARGUS_LINK_MODE}): ${ARGUS_LINK_LIBRARY}")
message(STATUS "Multimedia API include: ${MMAPI_INCLUDE_DIR}")
message(STATUS "nvv4l2: ${NVV4L2_LIBRARY}")
message(STATUS "nvbufsurface: ${NVBUFSURFACE_LIBRARY}")
