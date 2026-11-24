// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <notice_queue/notice_queue.hpp>

#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <utility>
#include <vector>

namespace
{

using camera_bridge::enqueue_notice;
using camera_bridge::FrameNotice;
using camera_bridge::QueuedNotice;

int g_failures = 0;

void check(bool condition, const char* what)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << what << "\n";
        ++g_failures;
    }
}

struct Fixture
{
    explicit Fixture(std::size_t slots) : ring_slots(slots)
    {
    }

    // Returns how many entries this notice dropped.
    std::size_t read(int slot, uint64_t generation)
    {
        FrameNotice n;
        n.slot = slot;
        n.slot_generation = generation;
        return enqueue_notice(queue, total_read, ring_slots, std::move(n));
    }

    // The publisher's ring: notice k (0-based) goes to slot k % ring_slots with generation k / ring_slots + 1.
    std::size_t read_sequential(uint64_t k)
    {
        return read(static_cast<int>(k % ring_slots), k / ring_slots + 1);
    }

    // Notices read after the front entry, as the worker computes it.
    uint64_t later_count() const
    {
        return total_read - queue.front().ordinal;
    }

    std::vector<uint64_t> ordinals() const
    {
        std::vector<uint64_t> out;
        for (const auto& q : queue)
            out.push_back(q.ordinal);
        return out;
    }

    std::deque<QueuedNotice> queue;
    uint64_t total_read = 0;
    std::size_t ring_slots;
};

void test_burst_in_distinct_slots()
{
    Fixture f(6);
    std::size_t dropped = f.read(0, 1) + f.read(1, 1) + f.read(2, 1);
    check(dropped == 0, "nothing is dropped");
    check((f.ordinals() == std::vector<uint64_t>{ 1, 2, 3 }), "all three survive in arrival order");
}

void test_burst_longer_than_ring()
{
    Fixture f(6);
    std::size_t dropped = 0;
    for (uint64_t k = 0; k < 10; ++k)
    {
        dropped += f.read_sequential(k);
        check(f.queue.size() <= f.ring_slots - 1, "queue never exceeds ring_slots - 1");
    }
    check(dropped == 5, "the five oldest are dropped");
    check((f.ordinals() == std::vector<uint64_t>{ 6, 7, 8, 9, 10 }), "survivors are the newest, in order");
}

void test_slot_reuse_keeps_newer_generation()
{
    Fixture f(6);
    std::size_t dropped = f.read(0, 1) + f.read(1, 1);
    check(f.read(0, 2) == 1, "reusing a slot drops the older generation");
    dropped += f.read(2, 1);
    check(dropped == 0, "other notices drop nothing");
    check((f.ordinals() == std::vector<uint64_t>{ 2, 3, 4 }), "only the newer generation of slot 0 remains");

    // An older generation arriving after a newer one never removes the newer.
    Fixture g(6);
    g.read(0, 2);
    check(g.read(0, 1) == 0 && g.queue.size() == 2 && g.queue.front().notice.slot_generation == 2,
          "a newer entry is never removed by an older notice");
}

void test_later_count_across_drops()
{
    Fixture f(6);
    for (uint64_t k = 0; k < 40; ++k)
    {
        f.read_sequential(k);
        check(f.later_count() == f.total_read - f.queue.front().ordinal, "later count tracks notices read after front");
        check(f.later_count() <= f.ring_slots - 2, "a kept front entry has at most ring_slots - 2 later notices");
    }

    // Slot reuse drops the front; its later count still counts the dropped notice.
    Fixture g(6);
    g.read(0, 1);
    g.read(1, 1);
    g.read(0, 2);
    g.read(2, 1);
    check(g.queue.front().ordinal == 2 && g.later_count() == 2, "later count includes a notice dropped after the front");

    // Boundary: ring_slots - 2 later notices is kept, ring_slots - 1 is treated as possibly overwritten.
    Fixture h(6);
    for (uint64_t k = 0; k < 5; ++k)
        h.read_sequential(k);
    check(h.queue.size() == 5 && h.later_count() == 4, "a later count of ring_slots - 2 is kept");
    check(h.read_sequential(5) == 1 && h.queue.front().ordinal == 2, "a later count of ring_slots - 1 is dropped");
}

void test_one_slot_ring_only_supersedes_same_slot()
{
    Fixture f(1);
    f.read(0, 1);
    check(f.read(0, 2) == 1 && f.queue.size() == 1 && f.queue.front().notice.slot_generation == 2,
          "the same slot is superseded");
    check(f.read(1, 1) + f.read(2, 1) == 0 && f.queue.size() == 3, "distinct slots are never dropped by ring depth");
}

} // namespace

int main()
{
    test_burst_in_distinct_slots();
    test_burst_longer_than_ring();
    test_slot_reuse_keeps_newer_generation();
    test_later_count_across_drops();
    test_one_slot_ring_only_supersedes_same_slot();
    if (g_failures == 0)
        std::cout << "all notice_queue tests passed\n";
    return g_failures == 0 ? 0 : 1;
}
