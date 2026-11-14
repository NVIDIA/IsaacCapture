# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# The messages between the camera plugin and its readers, as a FlatBuffer schema. The generated
# header is reached as <camera_wire_schema/camera_wire_generated.h>; only the codec and its
# tests include it.

include(${CMAKE_SOURCE_DIR}/cmake/GenerateFlatBuffers.cmake)

generate_flatbuffer_headers(
    CAMERA_WIRE_GENERATED_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}/generated/camera_wire_schema
    CAMERA_WIRE_GENERATED_BFBS
)

add_custom_target(camera_wire_schema_generation
    DEPENDS ${CAMERA_WIRE_GENERATED_HEADERS} ${CAMERA_WIRE_GENERATED_BFBS}
)

add_library(camera_wire_schema INTERFACE)
add_dependencies(camera_wire_schema camera_wire_schema_generation)

target_include_directories(camera_wire_schema INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
)

target_link_libraries(camera_wire_schema INTERFACE
    flatbuffers
)

add_library(camera::camera_wire_schema ALIAS camera_wire_schema)
