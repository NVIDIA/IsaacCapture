# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)

# The other process of the camera integration tests: it serves a camera group, or reads one
# and holds on to what it read, until it is killed.
add_executable(camera_test_peer
    main.cpp
)

target_link_libraries(camera_test_peer PRIVATE
    camera::camera_endpoint
    camera::camera_ipc
    camera::camera_reader
)
