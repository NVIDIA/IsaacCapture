# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==============================================================================
# Video stage: the Orin-to-PC video transport and the PC relay to LiveKit
# ==============================================================================
#   wire/    video_stage_wire   -- AVS1 records, clock exchange, H.264 access-unit helpers, frame correlator
#   sender/  video_stage_sender -- the Orin-side sender of AVS1 records over a direct link
#   app/     video_stage_worker -- the PC relay: receives the Orin's records and publishes them to LiveKit or a file
#
# No CUDA. The LiveKit SDK (src/livekit_publisher) is needed only by the relay.
#
# Configure with:
#   cmake -B build -DBUILD_VIDEO_STAGE=ON
# ==============================================================================

add_subdirectory(wire)
add_subdirectory(sender)
add_subdirectory(app)

if(BUILD_TESTING)
    add_subdirectory(wire/tests)
    add_subdirectory(tests)
endif()
