/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "FrameAlignmentPolicy.h"
#include "TestHarness.h"

#include <cstddef>
#include <vector>

int main()
{
    TestContext test("frame_alignment_policy");
    const uint64_t periodNs = 16666667ULL;
    const uint64_t halfPeriodNs = periodNs / 2;
    const uint64_t epochNs = 1000000000ULL;

    std::vector<argus_sender::FrameAlignmentSample> samples(4);
    for (size_t i = 0; i < samples.size(); ++i)
    {
        samples[i].valid = true;
        samples[i].framePeriodNs = periodNs;
        samples[i].sofTscNs = epochNs + i;
    }
    size_t index = 99;
    uint64_t spanNs = 0;
    TEST_CHECK_EQ(
        test, argus_sender::chooseFrameToReacquire(samples, &index, &spanNs), argus_sender::FRAME_ALIGNMENT_KEEP);
    TEST_CHECK_EQ(test, spanNs, 3ULL);

    samples[2].sofTscNs = epochNs - periodNs;
    TEST_CHECK_EQ(
        test, argus_sender::chooseFrameToReacquire(samples, &index, &spanNs), argus_sender::FRAME_ALIGNMENT_REACQUIRE);
    TEST_CHECK_EQ(test, index, static_cast<size_t>(2));

    samples[2].sofTscNs = epochNs - halfPeriodNs;
    samples[1].sofTscNs = epochNs;
    samples[3].sofTscNs = epochNs;
    TEST_CHECK_EQ(
        test, argus_sender::chooseFrameToReacquire(samples, &index, &spanNs), argus_sender::FRAME_ALIGNMENT_KEEP);

    samples[2].sofTscNs = epochNs - halfPeriodNs - 1;
    TEST_CHECK_EQ(
        test, argus_sender::chooseFrameToReacquire(samples, &index, &spanNs), argus_sender::FRAME_ALIGNMENT_REACQUIRE);
    TEST_CHECK_EQ(test, index, static_cast<size_t>(2));

    samples[0].valid = false;
    TEST_CHECK_EQ(test, argus_sender::chooseFrameToReacquire(samples, &index, &spanNs),
                  argus_sender::FRAME_ALIGNMENT_UNAVAILABLE);

    samples.clear();
    TEST_CHECK_EQ(
        test, argus_sender::chooseFrameToReacquire(samples, &index, &spanNs), argus_sender::FRAME_ALIGNMENT_KEEP);

    TEST_CHECK_EQ(test, argus_sender::kMaxFrameAlignmentDrops, static_cast<uint32_t>(4));

    std::vector<uint64_t> candidates;
    candidates.push_back(epochNs - 4000000ULL);
    candidates.push_back(epochNs + 3000000ULL);
    int64_t selectedDeltaNs = 0;
    TEST_CHECK(test, argus_sender::chooseClosestTscCandidate(candidates, epochNs, &index, &selectedDeltaNs));
    TEST_CHECK_EQ(test, index, static_cast<size_t>(1));
    TEST_CHECK_EQ(test, selectedDeltaNs, 3000000LL);

    candidates[0] = epochNs - 3000000ULL;
    TEST_CHECK(test, argus_sender::chooseClosestTscCandidate(candidates, epochNs, &index, &selectedDeltaNs));
    TEST_CHECK_EQ(test, index, static_cast<size_t>(1));
    TEST_CHECK_EQ(test, selectedDeltaNs, 3000000LL);

    candidates[0] = 0;
    candidates[1] = epochNs - 1000ULL;
    TEST_CHECK(test, argus_sender::chooseClosestTscCandidate(candidates, epochNs, &index, &selectedDeltaNs));
    TEST_CHECK_EQ(test, index, static_cast<size_t>(1));
    TEST_CHECK_EQ(test, selectedDeltaNs, -1000LL);

    candidates.assign(2, 0);
    TEST_CHECK(test, !argus_sender::chooseClosestTscCandidate(candidates, epochNs, &index, &selectedDeltaNs));
    TEST_CHECK(test, !argus_sender::chooseClosestTscCandidate(candidates, 0, &index, &selectedDeltaNs));

    std::vector<uint64_t> synchronizedTimestamps;
    synchronizedTimestamps.push_back(epochNs);
    synchronizedTimestamps.push_back(epochNs + 4000ULL);
    uint64_t referenceTscNs = 0;
    TEST_CHECK(test, argus_sender::computeSynchronizedTscReference(
                         synchronizedTimestamps, 1000000ULL, &referenceTscNs, &spanNs));
    TEST_CHECK_EQ(test, referenceTscNs, epochNs + 2000ULL);
    TEST_CHECK_EQ(test, spanNs, 4000ULL);

    synchronizedTimestamps[1] = epochNs + periodNs;
    TEST_CHECK(test, !argus_sender::computeSynchronizedTscReference(
                         synchronizedTimestamps, 1000000ULL, &referenceTscNs, &spanNs));
    TEST_CHECK_EQ(test, referenceTscNs, 0ULL);
    TEST_CHECK_EQ(test, spanNs, periodNs);

    synchronizedTimestamps[1] = 0;
    TEST_CHECK(test, !argus_sender::computeSynchronizedTscReference(
                         synchronizedTimestamps, 1000000ULL, &referenceTscNs, &spanNs));
    return test.finish();
}
