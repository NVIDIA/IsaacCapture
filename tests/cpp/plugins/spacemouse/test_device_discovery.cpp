// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "spacemouse_test_dirs.hpp"

#include <catch2/catch_test_macros.hpp>
#include <spacemouse/device_discovery.hpp>

#include <cstdint>
#include <vector>

using plugins::spacemouse::combined_report_from_descriptor;
using plugins::spacemouse::find_device;

namespace
{

// Multi-axis controller: report 1 X/Y/Z, report 2 Rx/Ry/Rz, report 3 buttons (SpaceMouse Compact).
const std::vector<uint8_t> kSeparateDescriptor = {
    0x05, 0x01, 0x09, 0x08, 0xA1, 0x01, // Generic Desktop, Multi-axis Controller, Application
    0xA1, 0x00, 0x85, 0x01, 0x16, 0xA2, 0xFE, 0x26, 0x5E, 0x01, // report 1, logical -350..350
    0x09, 0x30, 0x09, 0x31, 0x09, 0x32, 0x75, 0x10, 0x95, 0x03, 0x81, 0x06, 0xC0, // X Y Z
    0xA1, 0x00, 0x85, 0x02, 0x09, 0x33, 0x09, 0x34, 0x09, 0x35, 0x75, 0x10, 0x95, 0x03, 0x81, 0x06, 0xC0, // Rx Ry Rz
    0xA1, 0x02, 0x85, 0x03, 0x05, 0x09, 0x19, 0x01, 0x29, 0x02, 0x15, 0x00, 0x25, 0x01, // buttons 1-2
    0x75, 0x01, 0x95, 0x02, 0x81, 0x02, 0x95, 0x0E, 0x81, 0x03, 0xC0, // padding
    0xC0,
};

// Report 1 X/Y/Z/Rx/Ry/Rz, as a usage range (Universal Receiver, SpaceMouse Wireless BT).
const std::vector<uint8_t> kCombinedDescriptor = {
    0x05, 0x01, 0x09, 0x08, 0xA1, 0x01, 0xA1, 0x00, 0x85, 0x01, 0x16, 0xA2, 0xFE, 0x26,
    0x5E, 0x01, 0x19, 0x30, 0x29, 0x35, 0x75, 0x10, 0x95, 0x06, 0x81, 0x06, 0xC0, 0xC0,
};

// A vendor-defined interface with no motion axes (the receiver's other nodes).
const std::vector<uint8_t> kNoAxesDescriptor = {
    0x06, 0x00, 0xFF, 0x09, 0x01, 0xA1, 0x01, 0x85, 0x05, 0x15, 0x00, 0x26,
    0xFF, 0x00, 0x75, 0x08, 0x95, 0x07, 0x09, 0x01, 0x81, 0x02, 0xC0,
};

} // namespace

TEST_CASE("Discovery picks the first validated model in numeric hidraw order", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    devices.add("hidraw0", "Logitech USB Receiver");
    devices.add("hidraw10", "3Dconnexion SpaceMouse Compact");
    devices.add("hidraw9", "3Dconnexion SpaceNavigator for Notebooks");

    const auto found = find_device(devices.class_dir(), devices.dev_dir());

    REQUIRE(found);
    CHECK(found->path == devices.dev_dir() / "hidraw9");
    CHECK_FALSE(found->combined_report);
}

TEST_CASE("Without a report descriptor the report layout comes from the model", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    SECTION("Universal Receiver")
    {
        devices.add("hidraw4", "3Dconnexion Universal Receiver");
    }
    SECTION("SpaceMouse Wireless BT, not the older SpaceMouse Wireless")
    {
        devices.add("hidraw4", "3Dconnexion SpaceMouse Wireless BT");
    }

    const auto found = find_device(devices.class_dir(), devices.dev_dir());

    REQUIRE(found);
    CHECK(found->combined_report);
}

TEST_CASE("The report descriptor decides the report layout", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    // Named as a separate-report model, but its descriptor puts all six axes in report 1.
    devices.add("hidraw6", "3Dconnexion SpaceMouse Wireless", &kCombinedDescriptor);
    devices.add("hidraw7", "3Dconnexion SpaceMouse Compact", &kSeparateDescriptor);

    const auto found = find_device(devices.class_dir(), devices.dev_dir());

    REQUIRE(found);
    CHECK(found->path == devices.dev_dir() / "hidraw6");
    CHECK(found->combined_report);
}

TEST_CASE("Discovery skips a receiver's nodes that carry no motion axes", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    const char* receiver = "3Dconnexion 3Dconnexion Universal Receiver";
    devices.add("hidraw6", receiver, &kNoAxesDescriptor);
    devices.add("hidraw7", receiver, &kNoAxesDescriptor);
    devices.add("hidraw8", receiver, &kCombinedDescriptor);
    devices.add("hidraw10", receiver, &kNoAxesDescriptor);

    const auto found = find_device(devices.class_dir(), devices.dev_dir());

    REQUIRE(found);
    CHECK(found->path == devices.dev_dir() / "hidraw8");
    CHECK(found->combined_report);
}

TEST_CASE("Discovery finds nothing without a validated model or a hidraw class", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    devices.add("hidraw0", "Some Gamepad");
    devices.add("hidraw1", "3Dconnexion Universal Receiver", &kNoAxesDescriptor);

    CHECK_FALSE(find_device(devices.class_dir(), devices.dev_dir()));
    CHECK_FALSE(find_device(devices.class_dir() / "missing", devices.dev_dir()));
}

TEST_CASE("Report descriptors map to the layouts apply_report reads", "[spacemouse][discovery]")
{
    CHECK(combined_report_from_descriptor(kSeparateDescriptor) == false);
    CHECK(combined_report_from_descriptor(kCombinedDescriptor) == true);
    CHECK_FALSE(combined_report_from_descriptor(kNoAxesDescriptor));
    CHECK_FALSE(combined_report_from_descriptor(std::vector<uint8_t>{}));

    SECTION("4-byte usages carry their own usage page")
    {
        const std::vector<uint8_t> descriptor = {
            0x85, 0x01, 0x0B, 0x30, 0x00, 0x01, 0x00, 0x0B, 0x31, 0x00, 0x01, 0x00, 0x0B,
            0x32, 0x00, 0x01, 0x00, 0x0B, 0x33, 0x00, 0x01, 0x00, 0x0B, 0x34, 0x00, 0x01,
            0x00, 0x0B, 0x35, 0x00, 0x01, 0x00, 0x75, 0x10, 0x95, 0x06, 0x81, 0x06,
        };
        CHECK(combined_report_from_descriptor(descriptor) == true);
    }
    SECTION("Axes on another report ID or split differently are unsupported")
    {
        const std::vector<uint8_t> descriptor = {
            0x05, 0x01, 0x85, 0x04, 0x19, 0x30, 0x29, 0x35, 0x75, 0x10, 0x95, 0x06, 0x81, 0x06,
        };
        CHECK_FALSE(combined_report_from_descriptor(descriptor));
    }
    SECTION("A truncated item is malformed")
    {
        const std::vector<uint8_t> descriptor(kCombinedDescriptor.begin(), kCombinedDescriptor.begin() + 11);
        CHECK_FALSE(combined_report_from_descriptor(descriptor));
    }
}
