# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

add_library(pose_engine_stub STATIC
    stub_engine.cpp
    stub_engine.hpp
)

target_link_libraries(pose_engine_stub
    PUBLIC
        pose_provider::pose_engine_api
    PRIVATE
        Threads::Threads
)

add_library(pose_provider::pose_engine_stub ALIAS pose_engine_stub)
