# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(camera_livekit_bridge
    main.cpp
    stream_worker.cpp
)

target_link_libraries(camera_livekit_bridge
    PRIVATE
        camera_livekit_bridge::ipc_client
        camera_livekit_bridge::gpu_ops
        camera_livekit_bridge::nvenc
        camera_livekit_bridge::livekit_publisher
        camera_livekit_bridge::chrony_clock
        camera_livekit_bridge::video_types
        CUDA::cudart
        Threads::Threads
)

target_compile_options(camera_livekit_bridge PRIVATE -Wall -Wextra)
