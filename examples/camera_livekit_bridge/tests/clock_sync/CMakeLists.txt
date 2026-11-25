# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The header needs no SDK, but linking the publisher target pulls the SDK libraries in at link time.
add_executable(test_clock_sync test_clock_sync.cpp)
target_link_libraries(test_clock_sync PRIVATE camera_livekit_bridge::livekit_publisher)
add_test(NAME clock_sync COMMAND test_clock_sync)
