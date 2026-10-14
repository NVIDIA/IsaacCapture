// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "spacemouse_test_dirs.hpp"

#include <catch2/catch_test_macros.hpp>
#include <spacemouse/device_discovery.hpp>

using plugins::spacemouse::find_device;

TEST_CASE("Discovery picks the first validated model in hidraw order", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    devices.add("hidraw0", "Logitech USB Receiver");
    devices.add("hidraw2", "3Dconnexion SpaceMouse Compact");
    devices.add("hidraw1", "3Dconnexion SpaceNavigator for Notebooks");

    const auto found = find_device(devices.class_dir(), devices.dev_dir());

    REQUIRE(found);
    CHECK(found->path == devices.dev_dir() / "hidraw1");
    CHECK_FALSE(found->combined_report);
}

TEST_CASE("The Universal Receiver is discovered as a combined-report device", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    devices.add("hidraw4", "3Dconnexion Universal Receiver");

    const auto found = find_device(devices.class_dir(), devices.dev_dir());

    REQUIRE(found);
    CHECK(found->combined_report);
}

TEST_CASE("Discovery finds nothing without a validated model or a hidraw class", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    devices.add("hidraw0", "Some Gamepad");

    CHECK_FALSE(find_device(devices.class_dir(), devices.dev_dir()));
    CHECK_FALSE(find_device(devices.class_dir() / "missing", devices.dev_dir()));
}
