# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The PC worker: receives AVS1 access units from the Orin and hands them, untouched, to LiveKit or a file. With
# BUILD_VIDEO_STAGE_GPU it can also decode on NVDEC, apply a GPU operation and re-encode on NVENC first.
add_executable(video_stage_worker
    main.cpp
    control_channel.cpp
    livekit_output.cpp
    stream_worker.cpp
    tcp_receiver.cpp
)

target_link_libraries(video_stage_worker
    PRIVATE
        video_stage::wire
        sensing::livekit
)

if(BUILD_VIDEO_STAGE_GPU)
    target_sources(video_stage_worker PRIVATE pipeline.cpp)
    target_compile_definitions(video_stage_worker PRIVATE VIDEO_STAGE_GPU)
    target_link_libraries(video_stage_worker PRIVATE video_stage::core video_stage::codec)
endif()
