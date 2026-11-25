# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(ipc_client STATIC cuda_ipc_client.cpp)

target_include_directories(ipc_client
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

target_link_libraries(ipc_client
    PUBLIC
        nlohmann_json::nlohmann_json
    PRIVATE
        CUDA::cudart
)

target_compile_options(ipc_client PRIVATE -Wall -Wextra)

add_library(camera_livekit_bridge::ipc_client ALIAS ipc_client)
