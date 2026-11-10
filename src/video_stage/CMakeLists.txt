# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# Video stage: the Orin-to-PC video transport
# ==============================================================================
#   wire/    video_stage_wire   -- AVS1 records, clock exchange, H.264 access-unit helpers, frame correlator
#   sender/  video_stage_sender -- the Orin-side sender of AVS1 records over a direct link
#
# Plain C++ and POSIX sockets: no CUDA, no LiveKit.
#
# Configure with:
#   cmake -B build -DBUILD_VIDEO_STAGE=ON
# ==============================================================================

add_subdirectory(wire)
add_subdirectory(sender)

if(BUILD_TESTING)
    add_subdirectory(wire/tests)
endif()
