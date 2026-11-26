/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ScopedRelease.h"
#include "TestHarness.h"

#include <stdexcept>

namespace
{

struct FakeBuffer
{
    void Release()
    {
        ++releases;
    }
    int releases = 0;
};

// Holds one buffer and releases the previous one on replacement, like a sensor's pending frame.
struct PendingSlot
{
    void hold(FakeBuffer* buffer)
    {
        if (held)
            held->Release();
        held = buffer;
    }
    FakeBuffer* held = nullptr;
};

void failingFenceWait(FakeBuffer*)
{
    throw std::runtime_error("NvSciSyncFenceWait failed");
}

// Models one frameLoop iteration: acquire, fence wait, handoff.
void acquire(FakeBuffer* buffer, bool failFence, PendingSlot* pending)
{
    argus_sender::ScopedRelease<FakeBuffer> owned(buffer);
    if (failFence)
        failingFenceWait(buffer);
    pending->hold(buffer);
    owned.dismiss();
}

} // namespace

int main()
{
    TestContext test("scoped_release");

    {
        FakeBuffer buffer;
        {
            argus_sender::ScopedRelease<FakeBuffer> owned(&buffer);
            TEST_CHECK_EQ(test, buffer.releases, 0);
        }
        TEST_CHECK_EQ(test, buffer.releases, 1);
    }

    {
        FakeBuffer buffer;
        PendingSlot pending;
        bool caught = false;
        try
        {
            acquire(&buffer, true, &pending);
        }
        catch (const std::runtime_error&)
        {
            caught = true;
        }
        TEST_CHECK(test, caught);
        TEST_CHECK_EQ(test, buffer.releases, 1);
        TEST_CHECK(test, pending.held == nullptr);
    }

    {
        FakeBuffer buffer;
        PendingSlot pending;
        acquire(&buffer, false, &pending);
        TEST_CHECK_EQ(test, buffer.releases, 0);
        TEST_CHECK(test, pending.held == &buffer);

        FakeBuffer next;
        pending.hold(&next);
        TEST_CHECK_EQ(test, buffer.releases, 1);
        TEST_CHECK_EQ(test, next.releases, 0);
    }

    {
        FakeBuffer failed;
        FakeBuffer handedOff;
        PendingSlot pending;
        try
        {
            acquire(&failed, true, &pending);
        }
        catch (const std::runtime_error&)
        {
        }
        acquire(&handedOff, false, &pending);
        TEST_CHECK_EQ(test, failed.releases, 1);
        TEST_CHECK_EQ(test, handedOff.releases, 0);
        TEST_CHECK(test, pending.held == &handedOff);
    }

    return test.finish();
}
