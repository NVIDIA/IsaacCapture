# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The slot write-generation markers are shared with camera_receiver and its other readers; the header lives with them.
set(CAMERA_IPC_PROTOCOL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../camera_streamer/ipc_protocol"
    CACHE PATH "camera_streamer's ipc_protocol directory")
if(NOT EXISTS "${CAMERA_IPC_PROTOCOL_DIR}/inc/ipc_protocol/slot_markers.hpp")
    message(FATAL_ERROR "slot_markers.hpp not found under CAMERA_IPC_PROTOCOL_DIR=${CAMERA_IPC_PROTOCOL_DIR}")
endif()

add_library(ipc_client STATIC cuda_ipc_client.cpp)

target_include_directories(ipc_client
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

# The client's own sources include the markers header, and so do its consumers through cuda_ipc_client.hpp.
target_include_directories(ipc_client
    PUBLIC
        $<BUILD_INTERFACE:${CAMERA_IPC_PROTOCOL_DIR}/inc>
)

target_link_libraries(ipc_client
    PUBLIC
        nlohmann_json::nlohmann_json
        rt
    PRIVATE
        CUDA::cudart
)

target_compile_options(ipc_client PRIVATE -Wall -Wextra)

add_library(camera_livekit_bridge::ipc_client ALIAS ipc_client)
