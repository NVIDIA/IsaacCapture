# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The PC worker: receives AVS1 access units from the Orin, decodes on NVDEC, applies the GPU operation, re-encodes on
# NVENC and hands the access units to a sink.
add_executable(video_stage_worker
    main.cpp
    control_channel.cpp
    livekit_output.cpp
    pipeline.cpp
    stream_worker.cpp
    tcp_receiver.cpp
)

target_link_libraries(video_stage_worker
    PRIVATE
        video_stage::core
        video_stage::codec
        video_stage::wire
        sensing::livekit
)
