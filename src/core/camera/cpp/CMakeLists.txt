# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

find_package(Threads REQUIRED)

# The camera library; this module's README.md says what is in it. Linux only.
add_library(camera STATIC
    camera_host.cpp
    camera_plugin.cpp
    camera_reader.cpp
    codec.cpp
    config.cpp
    cuda_pool.cpp
    description.cpp
    endpoint.cpp
    endpoint_path.cpp
    host_pool.cpp
    message_socket.cpp
    options.cpp
    publisher.cpp
    inc/camera/camera_host.hpp
    inc/camera/camera_plugin.hpp
    inc/camera/camera_reader.hpp
    inc/camera/camera_source.hpp
    inc/camera/config.hpp
    inc/camera/cuda_pool.hpp
    inc/camera/endpoint.hpp
    inc/camera/endpoint_path.hpp
    inc/camera/host_pool.hpp
    inc/camera/message_socket.hpp
    inc/camera/messages.hpp
    inc/camera/options.hpp
    inc/camera/publisher.hpp
    inc/camera/types.hpp
    inc/camera/unique_fd.hpp
)

target_include_directories(camera INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(camera
    PRIVATE
        isaacteleop_schema
        log_bridge::log_bridge_core
        yaml-cpp::yaml-cpp
        Threads::Threads
)

# CUDA memory is built in when the CUDA Toolkit's headers are there. Nothing links CUDA: the
# driver's library is loaded when CUDA memory is first used, so the library and what links it
# run where no NVIDIA driver is installed.
if(CUDAToolkit_FOUND)
    target_compile_definitions(camera PRIVATE CAMERA_HAS_CUDA)
    target_include_directories(camera SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(camera PRIVATE ${CMAKE_DL_LIBS})
endif()

add_library(camera::camera ALIAS camera)
