/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CapturePriorityPolicy.h"
#include "TestHarness.h"

#include <vector>

int main()
{
    TestContext test("capture_priority_policy");

    argus_sender::AppConfig config;
    config.streams.resize(3);
    config.streams[0].type = argus_sender::STREAM_TYPE_MONO;
    config.streams[0].output.codec = argus_sender::CODEC_RAW_NV12;
    config.streams[1].type = argus_sender::STREAM_TYPE_STEREO;
    config.streams[1].output.codec = argus_sender::CODEC_HEVC;
    config.streams[2].type = argus_sender::STREAM_TYPE_MONO;
    config.streams[2].output.codec = argus_sender::CODEC_HEVC;

    std::vector<uint32_t> fps = { 60, 60, 30 };
    TEST_CHECK_EQ(test, argus_sender::selectPriorityStream(config, fps), static_cast<size_t>(1));

    fps[1] = 30;
    TEST_CHECK_EQ(test, argus_sender::selectPriorityStream(config, fps), static_cast<size_t>(0));

    TEST_CHECK_EQ(test, argus_sender::secondaryCaptureReleaseDelayNs(16666667ULL), 7000000ULL);
    TEST_CHECK_EQ(test, argus_sender::prioritySyncToleranceNs(16666667ULL), 8333333ULL);
    TEST_CHECK_EQ(test, argus_sender::prioritySyncToleranceNs(2000000ULL), 1000000ULL);

    return test.finish();
}
