# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The four-timestamp clock estimator, header only.
add_library(clock_exchange INTERFACE)

target_include_directories(clock_exchange
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
)

add_library(camera_livekit_bridge::clock_exchange ALIAS clock_exchange)
