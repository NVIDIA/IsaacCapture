// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <video_stage/encoded_video_frame.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("a default EncodedVideoFrame is empty and not a keyframe", "[EncodedVideoFrame]")
{
    const video_stage::EncodedVideoFrame frame;
    CHECK(frame.data == nullptr);
    CHECK(frame.size == 0);
    CHECK_FALSE(frame.keyframe);
    CHECK(frame.timestamp_us == 0);
    CHECK(frame.user_timestamp_us == 0);
    CHECK(frame.frame_id == 0);
    CHECK(frame.capture_tsc == 0);
}
