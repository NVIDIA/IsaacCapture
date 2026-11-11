/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "EncoderInputSlots.h"
#include "TestHarness.h"

#include <memory>
#include <string>
#include <utility>

namespace
{

// Appends "w<id>" when destroyed, so the log records the order in which leases and wrappers are dropped.
struct Wrapper
{
    Wrapper(int id, std::string* log) : id(id), log(log)
    {
    }
    ~Wrapper()
    {
        *log += "w" + std::to_string(id) + " ";
    }
    int id;
    std::string* log;
};

argus_sender::CaptureFrame leasedFrame(int id, std::string* log)
{
    argus_sender::CaptureFrame frame;
    frame.sequence = static_cast<uint64_t>(id);
    frame.lease = argus_sender::FrameLease([id, log]() { *log += "l" + std::to_string(id) + " "; });
    return frame;
}

void occupy(argus_sender::EncoderInputSlots<Wrapper>* slots, int id, std::string* log)
{
    slots->occupy(std::unique_ptr<Wrapper>(new Wrapper(id, log)), leasedFrame(id, log));
}

} // namespace

int main()
{
    TestContext test("encoder_input_slots");
    std::string log;

    {
        argus_sender::EncoderInputSlots<Wrapper> slots(2);
        TEST_CHECK(test, slots.hasFree());
        TEST_CHECK_EQ(test, slots.nextFree(), 0U);
        TEST_CHECK_EQ(test, slots.inFlight(), 0U);
        TEST_CHECK(test, !slots.complete(0)); // Nothing in flight.

        occupy(&slots, 1, &log);
        TEST_CHECK_EQ(test, slots.nextFree(), 1U);
        occupy(&slots, 2, &log);
        TEST_CHECK(test, !slots.hasFree());
        TEST_CHECK_EQ(test, slots.inFlight(), 2U);
        TEST_CHECK_EQ(test, log, std::string()); // Held while the encoder may read them.

        // The encoder may finish out of order; the lease goes back before the wrapper is destroyed.
        TEST_CHECK(test, slots.complete(1));
        TEST_CHECK_EQ(test, log, std::string("l2 w2 "));
        TEST_CHECK(test, !slots.complete(1)); // Already completed.
        TEST_CHECK(test, !slots.complete(2)); // Out of range.
        TEST_CHECK_EQ(test, slots.inFlight(), 1U);
        TEST_CHECK_EQ(test, slots.nextFree(), 1U);

        occupy(&slots, 3, &log);
        TEST_CHECK_EQ(test, slots.inFlight(), 2U);
        log.clear();
        slots.releaseAll();
        TEST_CHECK_EQ(test, log, std::string("l1 w1 l3 w3 "));
        TEST_CHECK_EQ(test, slots.inFlight(), 0U);
        TEST_CHECK(test, slots.hasFree());
        TEST_CHECK_EQ(test, slots.nextFree(), 0U);
        TEST_CHECK(test, !slots.complete(0));

        log.clear();
        occupy(&slots, 4, &log);
    }
    // Destroying the slots also drops the lease before the wrapper.
    TEST_CHECK_EQ(test, log, std::string("l4 w4 "));

    return test.finish();
}
