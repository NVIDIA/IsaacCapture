// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "spacemouse_test_dirs.hpp"

#include <catch2/catch_test_macros.hpp>
#include <spacemouse/device_discovery.hpp>

#include <initializer_list>

using plugins::spacemouse::find_device;

namespace
{

void write_descriptor(const spacemouse_test::FakeDevices& devices,
                      const char* name,
                      std::initializer_list<unsigned char> bytes)
{
    std::ofstream file(devices.class_dir() / name / "device" / "report_descriptor", std::ios::binary);
    for (const auto byte : bytes)
        file.put(static_cast<char>(byte));
}

} // namespace

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

TEST_CASE("Wireless BT is combined while the original Wireless uses separate reports", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    bool combined = false;
    SECTION("Wireless BT")
    {
        devices.add("hidraw4", "3Dconnexion SpaceMouse Wireless BT");
        combined = true;
    }
    SECTION("Original Wireless")
    {
        devices.add("hidraw4", "3Dconnexion SpaceMouse Wireless");
    }

    const auto found = find_device(devices.class_dir(), devices.dev_dir());

    REQUIRE(found);
    CHECK(found->combined_report == combined);
}

TEST_CASE("Receiver discovery selects motion regardless of node order", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    devices.add("hidraw10", "3Dconnexion 3Dconnexion Universal Receiver");
    devices.add("hidraw6", "3Dconnexion 3Dconnexion Universal Receiver");
    const char* motion = "hidraw6";
    const char* other = "hidraw10";
    SECTION("Motion sorts after the other interface")
    {
    }
    SECTION("Motion sorts before the other interface")
    {
        motion = "hidraw10";
        other = "hidraw6";
    }
    write_descriptor(devices, other, { 0x05, 0x01, 0x09, 0x02, 0xa1, 0x01, 0xc0 }); // Mouse collection
    write_descriptor(devices, motion, { 0x05, 0x01, 0x09, 0x08, 0xa1, 0x01, 0x85, 0x01, 0x16, 0xa2, 0xfe,
                                        0x26, 0x5e, 0x01, 0x09, 0x30, 0x09, 0x31, 0x09, 0x32, 0x09, 0x33,
                                        0x09, 0x34, 0x09, 0x35, 0x75, 0x10, 0x95, 0x06, 0x81, 0x02, 0xc0 });

    const auto found = find_device(devices.class_dir(), devices.dev_dir());

    REQUIRE(found);
    CHECK(found->path == devices.dev_dir() / motion);
    CHECK(found->combined_report);
}

TEST_CASE("Receiver discovery requires motion inputs in a readable descriptor", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    devices.add("hidraw4", "3Dconnexion Universal Receiver");
    bool expected = false;
    SECTION("Missing descriptor")
    {
    }
    SECTION("Truncated item")
    {
        write_descriptor(devices, "hidraw4", { 0x05, 0x01, 0x0b, 0x30 });
    }
    SECTION("Usage range on the button page")
    {
        write_descriptor(devices, "hidraw4", { 0x05, 0x09, 0x85, 0x01, 0x19, 0x30, 0x29, 0x35, 0x81, 0x02 });
    }
    SECTION("Axes declared only as output")
    {
        write_descriptor(devices, "hidraw4", { 0x05, 0x01, 0x85, 0x01, 0x19, 0x30, 0x29, 0x35, 0x91, 0x02, 0x81, 0x02 });
    }
    SECTION("Constant input")
    {
        write_descriptor(devices, "hidraw4", { 0x05, 0x01, 0x85, 0x01, 0x19, 0x30, 0x29, 0x35, 0x81, 0x01 });
    }
    SECTION("Motion usage range")
    {
        write_descriptor(devices, "hidraw4", { 0x05, 0x01, 0x85, 0x01, 0x19, 0x30, 0x29, 0x35, 0x81, 0x02 });
        expected = true;
    }
    SECTION("Translation and rotation in separate usage ranges on the same report")
    {
        write_descriptor(devices, "hidraw4", { 0x05, 0x01, 0x85, 0x01, 0x75, 0x10, 0x95, 0x03, 0x19, 0x30,
                                               0x29, 0x32, 0x81, 0x02, 0x19, 0x33, 0x29, 0x35, 0x81, 0x02 });
        expected = true;
    }
    CHECK(find_device(devices.class_dir(), devices.dev_dir()).has_value() == expected);
}

TEST_CASE("Receiver discovery skips reports the combined decoder cannot read", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    devices.add("hidraw10", "3Dconnexion Universal Receiver");
    SECTION("Axes only on report 4")
    {
        write_descriptor(devices, "hidraw10",
                         { 0x05, 0x01, 0x85, 0x04, 0x75, 0x10, 0x95, 0x06, 0x19, 0x30, 0x29, 0x35, 0x81, 0x02 });
    }
    SECTION("Translation on report 1 and rotation on report 2")
    {
        write_descriptor(devices, "hidraw10", { 0x05, 0x01, 0x85, 0x01, 0x75, 0x10, 0x95, 0x03, 0x19, 0x30, 0x29,
                                                0x32, 0x81, 0x02, 0x85, 0x02, 0x19, 0x33, 0x29, 0x35, 0x81, 0x02 });
    }
    SECTION("Unnumbered motion report")
    {
        write_descriptor(devices, "hidraw10", { 0x05, 0x01, 0x75, 0x10, 0x95, 0x06, 0x19, 0x30, 0x29, 0x35, 0x81, 0x02 });
    }
    CHECK_FALSE(find_device(devices.class_dir(), devices.dev_dir()));

    devices.add("hidraw6", "3Dconnexion Universal Receiver");
    write_descriptor(
        devices, "hidraw6", { 0x05, 0x01, 0x85, 0x01, 0x75, 0x10, 0x95, 0x06, 0x19, 0x30, 0x29, 0x35, 0x81, 0x02 });
    const auto found = find_device(devices.class_dir(), devices.dev_dir());
    REQUIRE(found);
    CHECK(found->path == devices.dev_dir() / "hidraw6");
    CHECK(found->combined_report);
}

TEST_CASE("Discovery finds nothing without a validated model or a hidraw class", "[spacemouse][discovery]")
{
    spacemouse_test::FakeDevices devices;
    devices.add("hidraw0", "Some Gamepad");

    CHECK_FALSE(find_device(devices.class_dir(), devices.dev_dir()));
    CHECK_FALSE(find_device(devices.class_dir() / "missing", devices.dev_dir()));
}
