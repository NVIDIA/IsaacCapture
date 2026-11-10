# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The PC relay: receives AVS1 access units from the Orin over its direct link and hands them, untouched, to LiveKit or
# a file.
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
