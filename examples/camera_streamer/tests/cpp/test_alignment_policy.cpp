/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "StereoAlignmentPolicy.h"
#include "TestHarness.h"

int main()
{
    TestContext test("alignment_policy");
    const uint64_t framePeriodNs = 16666667ULL;
    const uint64_t halfFrameNs = framePeriodNs / 2;

    argus_sender::FrameTiming left;
    argus_sender::FrameTiming right;
    left.hasSensorSofTimestampTsc = true;
    right.hasSensorSofTimestampTsc = true;
    left.sensorSofTimestampTscNs = 1000000000ULL;

    int64_t deltaNs = 0;
    right.sensorSofTimestampTscNs = left.sensorSofTimestampTscNs + halfFrameNs;
    TEST_CHECK_EQ(test, argus_sender::chooseStereoAlignmentAction(left, right, framePeriodNs, 0, &deltaNs),
                  argus_sender::STEREO_ALIGNMENT_KEEP);
    TEST_CHECK_EQ(test, deltaNs, static_cast<int64_t>(halfFrameNs));

    right.sensorSofTimestampTscNs = left.sensorSofTimestampTscNs + halfFrameNs + 1;
    TEST_CHECK_EQ(test, argus_sender::chooseStereoAlignmentAction(left, right, framePeriodNs, 0),
                  argus_sender::STEREO_ALIGNMENT_DROP_LEFT);

    right.sensorSofTimestampTscNs = left.sensorSofTimestampTscNs - halfFrameNs - 1;
    TEST_CHECK_EQ(test, argus_sender::chooseStereoAlignmentAction(left, right, framePeriodNs, 0),
                  argus_sender::STEREO_ALIGNMENT_DROP_RIGHT);

    TEST_CHECK_EQ(
        test,
        argus_sender::chooseStereoAlignmentAction(left, right, framePeriodNs, argus_sender::kStereoMaxAlignmentDrops),
        argus_sender::STEREO_ALIGNMENT_KEEP);

    TEST_CHECK_EQ(
        test, argus_sender::chooseStereoAlignmentAction(left, right, 0, 0), argus_sender::STEREO_ALIGNMENT_KEEP);

    left.hasSensorSofTimestampTsc = false;
    right.hasSensorSofTimestampTsc = false;
    left.hasSensorTimestamp = true;
    right.hasSensorTimestamp = true;
    left.sensorTimestampNs = 2000000000ULL;
    right.sensorTimestampNs = left.sensorTimestampNs + halfFrameNs + 1;
    TEST_CHECK_EQ(test, argus_sender::chooseStereoAlignmentAction(left, right, framePeriodNs, 0),
                  argus_sender::STEREO_ALIGNMENT_DROP_LEFT);

    left.hasSensorSofTimestampTsc = true;
    right.hasSensorSofTimestampTsc = true;
    left.sensorSofTimestampTscNs = 3000000000ULL;
    right.sensorSofTimestampTscNs = left.sensorSofTimestampTscNs + 1;
    TEST_CHECK_EQ(test, argus_sender::chooseStereoAlignmentAction(left, right, framePeriodNs, 0),
                  argus_sender::STEREO_ALIGNMENT_KEEP);

    left.hasSensorSofTimestampTsc = false;
    right.hasSensorSofTimestampTsc = false;
    left.hasSensorTimestamp = false;
    right.hasSensorTimestamp = false;
    TEST_CHECK(test, !argus_sender::stereoAlignmentDeltaNs(left, right, &deltaNs));
    TEST_CHECK_EQ(test, argus_sender::chooseStereoAlignmentAction(left, right, framePeriodNs, 0),
                  argus_sender::STEREO_ALIGNMENT_KEEP);

    return test.finish();
}
