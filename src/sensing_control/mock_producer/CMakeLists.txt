# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# A camera producer without cameras: it plays H.264 files through the same management channel, supervisor and direct
# sender as camera_plugin_sensing, so the PC side can be exercised with no Orin.
add_executable(sensing_mock_producer main.cpp)

target_link_libraries(sensing_mock_producer
    PRIVATE
        sensing::control
        video_stage::sender
        video_stage::wire
)
