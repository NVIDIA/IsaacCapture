// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "xdev_hand_tracker_selection.hpp"

#include <catch2/catch_test_macros.hpp>

using core::detail::classify_xdev_hand_tracker;
using core::detail::XDevHandTrackerPriority;

TEST_CASE("XDev hand tracker selection prioritizes explicit push devices", "[deviceio][xdev][hand]")
{
    const auto left = classify_xdev_hand_tracker("Push Hand Tracker (Left)", "Push-L");
    CHECK(left.priority == XDevHandTrackerPriority::preferred);
    CHECK(left.left);
    CHECK_FALSE(left.right);

    const auto right = classify_xdev_hand_tracker("Push Hand Tracker (Right)", "Push-R");
    CHECK(right.priority == XDevHandTrackerPriority::preferred);
    CHECK_FALSE(right.left);
    CHECK(right.right);
}

TEST_CASE("XDev hand tracker selection keeps head-device fallbacks", "[deviceio][xdev][hand]")
{
    const auto generic = classify_xdev_hand_tracker("Head Device", "N/A");
    CHECK(generic.priority == XDevHandTrackerPriority::display_fallback);
    CHECK(generic.left);
    CHECK(generic.right);

    const auto left = classify_xdev_hand_tracker("Left Head Device Hand", "Head Device (0)");
    CHECK(left.priority == XDevHandTrackerPriority::display_fallback);
    CHECK(left.left);
    CHECK_FALSE(left.right);

    const auto right = classify_xdev_hand_tracker("Right Head Device Hand", "Head Device (1)");
    CHECK(right.priority == XDevHandTrackerPriority::display_fallback);
    CHECK_FALSE(right.left);
    CHECK(right.right);
}

TEST_CASE("XDev hand tracker selection defers unrelated controller profiles", "[deviceio][xdev][hand]")
{
    for (const char* name :
         { "DebugController Left", "Quest Touch Plus Right", "Pico 4 Ultra Left", "PS Sense Right", "Xbox Controller" })
    {
        const auto selection = classify_xdev_hand_tracker(name, "N/A");
        CHECK(selection.priority == XDevHandTrackerPriority::unclassified_fallback);
    }
}
