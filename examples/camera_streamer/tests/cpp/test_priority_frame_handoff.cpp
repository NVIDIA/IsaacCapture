/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "PriorityFrameHandoff.h"
#include "TestHarness.h"

#include <memory>

namespace
{

struct Frame
{
    Frame(int number, int* released) : number(number), released(released)
    {
    }
    ~Frame()
    {
        ++*released;
    }
    int number;
    int* released;
};

} // namespace

int main()
{
    TestContext test("priority_frame_handoff");
    argus_sender::LatestFrameSlot<Frame> slot;
    int released = 0;

    TEST_CHECK(test, !slot.occupied());
    TEST_CHECK(test, !slot.take());
    TEST_CHECK(test, !slot.replace(std::unique_ptr<Frame>(new Frame(1, &released))));
    TEST_CHECK(test, slot.occupied());

    std::unique_ptr<Frame> stale = slot.replace(std::unique_ptr<Frame>(new Frame(2, &released)));
    TEST_CHECK_EQ(test, stale->number, 1);
    TEST_CHECK_EQ(test, released, 0);
    stale.reset(); // Release outside the producer's slot lock.
    TEST_CHECK_EQ(test, released, 1);

    std::unique_ptr<Frame> newest = slot.take();
    TEST_CHECK_EQ(test, newest->number, 2);
    TEST_CHECK(test, !slot.occupied());
    newest.reset();
    TEST_CHECK_EQ(test, released, 2);

    slot.replace(std::unique_ptr<Frame>(new Frame(3, &released)));
    slot.reset(); // Shutdown releases any unconsumed frame.
    TEST_CHECK_EQ(test, released, 3);
    TEST_CHECK(test, !slot.occupied());

    const uint64_t start = 10000000000ULL;
    TEST_CHECK(test, !argus_sender::priorityFrameDroughtExpired(
                         start, start + argus_sender::kPrioritySteadyDroughtNs - 1, true));
    TEST_CHECK(
        test, argus_sender::priorityFrameDroughtExpired(start, start + argus_sender::kPrioritySteadyDroughtNs, true));
    TEST_CHECK(test, !argus_sender::priorityFrameDroughtExpired(
                         start, start + argus_sender::kPriorityStartupDroughtNs - 1, false));
    TEST_CHECK(
        test, argus_sender::priorityFrameDroughtExpired(start, start + argus_sender::kPriorityStartupDroughtNs, false));
    TEST_CHECK(test, !argus_sender::priorityFrameDroughtExpired(start, start - 1, true));

    argus_sender::PriorityFrameProgress progress(start);
    TEST_CHECK_EQ(test, progress.droughtLimitNs(), argus_sender::kPriorityStartupDroughtNs);
    TEST_CHECK(test, !progress.droughtExpired(start + argus_sender::kPriorityStartupDroughtNs - 1));
    progress.noteFreshFrame(start + 100);
    TEST_CHECK_EQ(test, progress.droughtLimitNs(), argus_sender::kPrioritySteadyDroughtNs);
    TEST_CHECK(test, !progress.droughtExpired(start + 100 + argus_sender::kPrioritySteadyDroughtNs - 1));
    TEST_CHECK(test, progress.droughtExpired(start + 100 + argus_sender::kPrioritySteadyDroughtNs));
    progress.noteFreshFrame(start + 200 + argus_sender::kPrioritySteadyDroughtNs);
    TEST_CHECK(test, !progress.droughtExpired(start + 200 + argus_sender::kPrioritySteadyDroughtNs));

    return test.finish();
}
