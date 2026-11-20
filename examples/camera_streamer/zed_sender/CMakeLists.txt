# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
find_program(CMAKE_CUDA_COMPILER nvcc HINTS /usr/local/cuda/bin REQUIRED)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 110)
endif()
project(zed_sender LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)
find_package(Threads REQUIRED)

set(MMAPI_ROOT "/usr/src/jetson_multimedia_api" CACHE PATH "Jetson Multimedia API")
set(MMAPI_CLASSES "${MMAPI_ROOT}/samples/common/classes")
set(COMMON "${CMAKE_CURRENT_SOURCE_DIR}/../argus_sender/src")
find_package(yaml-cpp CONFIG QUIET)
if(NOT yaml-cpp_FOUND)
    include(FetchContent)
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    set(YAML_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(yaml_cpp
        URL https://github.com/jbeder/yaml-cpp/releases/download/yaml-cpp-0.9.0/yaml-cpp-yaml-cpp-0.9.0.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        URL_HASH SHA256=298593d9c440fd9034b8b193d96318b76d49bc97c6ceadb7b0836edf0b6d7539)
    FetchContent_MakeAvailable(yaml_cpp)
endif()

foreach(lib nvv4l2 nvbufsurface nvbufsurftransform)
    find_library(${lib}_LIBRARY NAMES ${lib}
        HINTS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/nvidia /usr/lib/aarch64-linux-gnu/nvidia)
    if(NOT ${lib}_LIBRARY)
        message(FATAL_ERROR "${lib} not found: install the matching Jetson Multimedia API")
    endif()
endforeach()

set(MMAPI_SOURCES
    ${MMAPI_CLASSES}/NvBuffer.cpp
    ${MMAPI_CLASSES}/NvElement.cpp
    ${MMAPI_CLASSES}/NvElementProfiler.cpp
    ${MMAPI_CLASSES}/NvLogging.cpp
    ${MMAPI_CLASSES}/NvV4l2Element.cpp
    ${MMAPI_CLASSES}/NvV4l2ElementPlane.cpp
    ${MMAPI_CLASSES}/NvVideoEncoder.cpp)

add_executable(zed_sender
    src/main.cpp src/Config.cpp src/V4l2Capture.cpp src/HardwareEncoder.cpp src/CudaConversion.cu
    ${COMMON}/Avs1Wire.cpp ${COMMON}/TcpTransport.cpp ${COMMON}/Runtime.cpp ${COMMON}/Clock.cpp
    ${MMAPI_SOURCES})
target_include_directories(zed_sender PRIVATE src ${COMMON} ${MMAPI_ROOT}/include /usr/local/cuda/include)
target_compile_definitions(zed_sender PRIVATE NVMMAPI_SUPPORTED)
target_compile_options(zed_sender PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)
set_source_files_properties(${MMAPI_SOURCES} PROPERTIES COMPILE_OPTIONS -w)
target_link_libraries(zed_sender PRIVATE yaml-cpp::yaml-cpp Threads::Threads CUDA::cudart
    ${nvv4l2_LIBRARY} ${nvbufsurface_LIBRARY} ${nvbufsurftransform_LIBRARY})
install(TARGETS zed_sender DESTINATION bin)

if(BUILD_TESTING)
    add_executable(zed_config_test tests/test_config.cpp src/Config.cpp)
    target_include_directories(zed_config_test PRIVATE src)
    target_link_libraries(zed_config_test PRIVATE yaml-cpp::yaml-cpp)
    add_test(NAME zed_config COMMAND zed_config_test)
endif()

# Explicit hardware test target; never runs as part of CTest or normal builds.
add_executable(zed_encoder_pattern EXCLUDE_FROM_ALL
    tests/encoder_pattern.cpp src/HardwareEncoder.cpp src/CudaConversion.cu ${MMAPI_SOURCES})
target_include_directories(zed_encoder_pattern PRIVATE src ${MMAPI_ROOT}/include /usr/local/cuda/include)
target_link_libraries(zed_encoder_pattern PRIVATE Threads::Threads CUDA::cudart
    ${nvv4l2_LIBRARY} ${nvbufsurface_LIBRARY} ${nvbufsurftransform_LIBRARY})
