# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(camera_ipc_probe main.cpp)

target_link_libraries(camera_ipc_probe
    PRIVATE
        camera_livekit_bridge::ipc_client
        CUDA::cudart
)

target_compile_options(camera_ipc_probe PRIVATE -Wall -Wextra)
