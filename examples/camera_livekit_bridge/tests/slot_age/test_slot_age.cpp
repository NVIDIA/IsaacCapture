// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <video_types/slot_age.hpp>

#include <cstdint>
#include <iostream>

namespace
{

int g_failures = 0;

void check(bool condition, const char* what)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << what << "\n";
        ++g_failures;
    }
}

} // namespace

int main()
{
    using camera_bridge::classify_slot_age;
    using camera_bridge::slot_read_deadline_ns;
    using camera_bridge::SlotAge;

    // The publisher wraps a ring of N slots after N - 1 further frames.
    check(slot_read_deadline_ns(30, 6) == 5 * 1'000'000'000LL / 30, "6 slots at 30 fps");
    check(slot_read_deadline_ns(30, 6) >= 166'666'666 && slot_read_deadline_ns(30, 6) <= 166'666'667,
          "6 slots at 30 fps is about 166.67 ms");
    check(slot_read_deadline_ns(60, 3) == 2 * 1'000'000'000LL / 60, "3 slots at 60 fps");
    check(slot_read_deadline_ns(30, 2) == 1'000'000'000LL / 30, "2 slots is the smallest usable ring");
    check(slot_read_deadline_ns(0, 6) == 0, "fps 0 disables the check");
    check(slot_read_deadline_ns(30, 1) == 0, "a one-slot ring disables the check");
    check(slot_read_deadline_ns(30, 0) == 0, "an empty ring disables the check");

    const int64_t now = 10'000'000'000;
    const int64_t deadline = 166'000'000;
    const uint64_t notice = static_cast<uint64_t>(now - 100'000'000);

    check(classify_slot_age(notice, now, deadline) == SlotAge::Fresh, "well within the deadline is fresh");
    check(classify_slot_age(static_cast<uint64_t>(now), now, deadline) == SlotAge::Fresh, "age 0 is fresh");
    check(classify_slot_age(static_cast<uint64_t>(now - deadline), now, deadline) == SlotAge::Fresh,
          "exactly at the deadline is fresh");
    check(classify_slot_age(static_cast<uint64_t>(now - deadline - 1), now, deadline) == SlotAge::Stale,
          "one nanosecond past the deadline is stale");
    check(classify_slot_age(static_cast<uint64_t>(now - 5'000'000'000), now, deadline) == SlotAge::Stale,
          "long past the deadline is stale");

    check(classify_slot_age(0, now, deadline) == SlotAge::Unknown, "no notice time is unknown");
    check(classify_slot_age(static_cast<uint64_t>(now + 1), now, deadline) == SlotAge::Unknown,
          "a notice from the future is unknown");
    check(classify_slot_age(static_cast<uint64_t>(now + 5'000'000'000), now, deadline) == SlotAge::Unknown,
          "a notice far in the future is unknown");
    check(classify_slot_age(notice, now, 0) == SlotAge::Unknown, "deadline 0 disables the check");
    check(classify_slot_age(notice, now, -1) == SlotAge::Unknown, "a negative deadline disables the check");
    // Even a very old notice is not judged when the check is off.
    check(classify_slot_age(1, now, 0) == SlotAge::Unknown, "deadline 0 beats an ancient notice");

    if (g_failures != 0)
    {
        std::cerr << g_failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "slot_age: all checks passed\n";
    return 0;
}
