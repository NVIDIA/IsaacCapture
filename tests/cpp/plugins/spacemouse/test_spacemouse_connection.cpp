// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The connection reads a FIFO standing in for /dev/hidrawN: writing reports into it feeds read(),
// and closing the last writer makes read() return end of file, which is how an unplug looks.

#include "spacemouse_test_dirs.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <spacemouse/spacemouse_connection.hpp>

#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <unistd.h>
#include <vector>

using plugins::spacemouse::ConnectionStatus;
using plugins::spacemouse::SpaceMouseConnection;
using plugins::spacemouse::SpaceMouseConnectionConfig;

namespace
{

constexpr int64_t kSecond = 1'000'000'000;

SpaceMouseConnectionConfig discovering(const spacemouse_test::FakeDevices& devices)
{
    SpaceMouseConnectionConfig config;
    config.hidraw_class_dir = devices.class_dir();
    config.dev_dir = devices.dev_dir();
    return config;
}

//! The device side of a FIFO node: opened read-write so the plugin's reader never sees EOF early.
class DeviceWriter
{
public:
    explicit DeviceWriter(const std::filesystem::path& node) : fd_(::open(node.c_str(), O_RDWR | O_NONBLOCK))
    {
        REQUIRE(fd_ >= 0);
    }
    ~DeviceWriter()
    {
        unplug();
    }
    void send(const std::vector<uint8_t>& report)
    {
        REQUIRE(::write(fd_, report.data(), report.size()) == static_cast<ssize_t>(report.size()));
    }
    void unplug()
    {
        if (fd_ >= 0)
            ::close(fd_);
        fd_ = -1;
    }

private:
    int fd_;
};

} // namespace

TEST_CASE("No SpaceMouse connected: disconnected at rest", "[spacemouse][connection]")
{
    spacemouse_test::FakeDevices devices;
    SpaceMouseConnection connection(discovering(devices));

    connection.poll(0);

    CHECK(connection.status() == ConnectionStatus::NoDevice);
    CHECK(connection.state().translation == std::array<float, 3>{});
}

TEST_CASE("A discovered device is read until it is unplugged", "[spacemouse][connection]")
{
    spacemouse_test::FakeDevices devices;
    DeviceWriter device(devices.add("hidraw0", "3Dconnexion SpaceMouse Compact"));
    SpaceMouseConnection connection(discovering(devices));

    connection.poll(0);
    REQUIRE(connection.status() == ConnectionStatus::Connected);

    // A FIFO keeps no report boundaries, so send one report per poll.
    device.send({ 1, 0x5E, 0x01, 0, 0, 0, 0 }); // translation x = 350
    connection.poll(1);
    device.send({ 3, 0b01 });
    connection.poll(2);
    CHECK(connection.state().translation[0] == Catch::Approx(1.0f));
    CHECK(connection.state().buttons[0] == 1);

    device.unplug();
    connection.poll(3);
    CHECK(connection.status() == ConnectionStatus::NoDevice);
    CHECK(connection.state().translation == std::array<float, 3>{});
    CHECK(connection.state().buttons[0] == 0);
}

TEST_CASE("A missing device is retried at most once per retry interval", "[spacemouse][connection]")
{
    spacemouse_test::FakeDevices devices;
    SpaceMouseConnection connection(discovering(devices));
    connection.poll(0);
    REQUIRE(connection.status() == ConnectionStatus::NoDevice);

    DeviceWriter device(devices.add("hidraw0", "3Dconnexion SpaceMouse Wireless"));
    connection.poll(kSecond / 2);
    CHECK(connection.status() == ConnectionStatus::NoDevice);

    connection.poll(kSecond);
    CHECK(connection.status() == ConnectionStatus::Connected);
}

TEST_CASE("An explicit device path that does not exist is no device", "[spacemouse][connection]")
{
    spacemouse_test::FakeDevices devices;
    SpaceMouseConnectionConfig config;
    config.device_path = devices.dev_dir() / "hidraw9";
    SpaceMouseConnection connection(config);

    connection.poll(0);

    CHECK(connection.status() == ConnectionStatus::NoDevice);
    CHECK(connection.error().empty());
}

TEST_CASE("A device that cannot be opened reports why", "[spacemouse][connection]")
{
    if (::geteuid() == 0)
        SKIP("root can open any file");

    spacemouse_test::FakeDevices devices;
    const auto node = devices.dev_dir() / "hidraw0";
    std::ofstream(node).put('\0');
    std::filesystem::permissions(node, std::filesystem::perms::none);
    SpaceMouseConnectionConfig config;
    config.device_path = node;
    SpaceMouseConnection connection(config);

    connection.poll(0);

    CHECK(connection.status() == ConnectionStatus::OpenFailed);
    CHECK(connection.error().find(node.string()) != std::string::npos);
}

TEST_CASE("A held deflection keeps its axes while the device keeps reporting", "[spacemouse][connection]")
{
    spacemouse_test::FakeDevices devices;
    DeviceWriter device(devices.add("hidraw0", "3Dconnexion SpaceMouse Compact"));
    SpaceMouseConnectionConfig config = discovering(devices);
    SpaceMouseConnection connection(config);
    connection.poll(0);

    // The device repeats an unchanged report every 16 ms for as long as it is held.
    constexpr int64_t kReportInterval = 16'000'000;
    for (int64_t now = 0; now < 4 * config.stale_axes_timeout_ns; now += kReportInterval)
    {
        device.send({ 1, 0x5E, 0x01, 0, 0, 0, 0 });
        connection.poll(now);
    }
    CHECK(connection.state().translation[0] == Catch::Approx(1.0f));
}

TEST_CASE("Deflected axes are zeroed once an open device goes silent", "[spacemouse][connection]")
{
    spacemouse_test::FakeDevices devices;
    DeviceWriter device(devices.add("hidraw0", "3Dconnexion SpaceMouse Wireless"));
    SpaceMouseConnectionConfig config = discovering(devices);
    SpaceMouseConnection connection(config);
    connection.poll(0);

    device.send({ 2, 0, 0, 0, 0, 0x5E, 0x01 }); // rotation z = 350
    connection.poll(1);
    device.send({ 3, 0b01 });
    connection.poll(2);
    connection.poll(2 + config.stale_axes_timeout_ns - 1);
    CHECK(connection.state().rotation[2] == Catch::Approx(1.0f));

    connection.poll(2 + config.stale_axes_timeout_ns);
    CHECK(connection.status() == ConnectionStatus::Connected);
    CHECK(connection.state().rotation == std::array<float, 3>{});
    CHECK(connection.state().buttons[0] == 1);
}
