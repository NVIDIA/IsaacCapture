// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <avatar/avatar_health.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

using plugins::avatar::GloveHealthInput;
using plugins::avatar::UsbDevicePresence;

namespace
{

class TemporaryDirectory
{
public:
    TemporaryDirectory()
    {
        const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / ("sharpa-avatar-health-" + std::to_string(nonce));
        std::filesystem::create_directories(path);
    }

    ~TemporaryDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }

    std::filesystem::path path;
};

void write_text(const std::filesystem::path& path, const std::string& value)
{
    std::ofstream stream(path);
    REQUIRE(stream);
    stream << value;
    REQUIRE(stream.good());
}

} // anonymous namespace

TEST_CASE("Glove health preserves startup, stream, and configuration states", "[sharpa_avatar][health]")
{
    GloveHealthInput input;

    SECTION("disabled dataset")
    {
        input.enabled = false;
        const auto status = plugins::avatar::make_glove_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_DISABLED);
        CHECK(status.reason == core::PluginDeviceReason_DISABLED_BY_CONFIGURATION);
    }

    SECTION("startup before discovery")
    {
        const auto status = plugins::avatar::make_glove_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_UNKNOWN);
        CHECK(status.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);
    }

    SECTION("previously observed glove disappears")
    {
        input.ever_observed = true;
        const auto status = plugins::avatar::make_glove_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_DISCONNECTED);
        CHECK(status.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);
    }

    SECTION("SDK online false is a data condition")
    {
        input.inventory_present = true;
        input.ever_observed = true;
        input.ever_received_fresh_data = true;
        const auto status = plugins::avatar::make_glove_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_DEGRADED);
        CHECK(status.reason == core::PluginDeviceReason_NO_CURRENT_DATA);
        CHECK(status.error.find("online=false") != std::string::npos);
    }

    SECTION("SDK offline before first stream remains startup waiting")
    {
        input.inventory_present = true;
        input.ever_observed = true;
        const auto status = plugins::avatar::make_glove_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_UNKNOWN);
        CHECK(status.reason == core::PluginDeviceReason_NO_CURRENT_DATA);
    }

    SECTION("SDK online waits for the first fresh sample")
    {
        input.inventory_present = true;
        input.ever_observed = true;
        input.sdk_online = true;
        const auto status = plugins::avatar::make_glove_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_UNKNOWN);
        CHECK(status.reason == core::PluginDeviceReason_NO_CURRENT_DATA);
    }

    SECTION("fresh data and SDK online are healthy")
    {
        input.inventory_present = true;
        input.ever_observed = true;
        input.sdk_online = true;
        input.fresh_data = true;
        input.ever_received_fresh_data = true;
        const auto status = plugins::avatar::make_glove_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_CONNECTED);
        CHECK(status.reason == core::PluginDeviceReason_NONE);
        CHECK(status.error.empty());
    }

    SECTION("stale data after streaming is degraded")
    {
        input.inventory_present = true;
        input.ever_observed = true;
        input.sdk_online = true;
        input.ever_received_fresh_data = true;
        const auto status = plugins::avatar::make_glove_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_DEGRADED);
        CHECK(status.reason == core::PluginDeviceReason_NO_CURRENT_DATA);
        CHECK(status.error.find("3-second freshness window") != std::string::npos);
    }

    SECTION("published paths can report independent freshness")
    {
        input.inventory_present = true;
        input.ever_observed = true;
        input.sdk_online = true;
        input.ever_received_fresh_data = true;

        input.fresh_data = true;
        const auto human = plugins::avatar::make_glove_status("/hand/left", input);
        input.fresh_data = false;
        const auto raw = plugins::avatar::make_glove_status("/glove/avatar_raw_left", input);

        CHECK(human.state == core::PluginDeviceState_CONNECTED);
        CHECK(raw.state == core::PluginDeviceState_DEGRADED);
        CHECK(raw.reason == core::PluginDeviceReason_NO_CURRENT_DATA);
    }
}

TEST_CASE("Dongle status reports Linux USB enumeration", "[sharpa_avatar][health]")
{
    TemporaryDirectory sysfs;

    CHECK(plugins::avatar::detect_usb_device_presence(sysfs.path, 0x1915, 0x520F) == UsbDevicePresence::ABSENT);

    const auto dongle = sysfs.path / "1-2";
    std::filesystem::create_directories(dongle);
    write_text(dongle / "idVendor", "1915\n");
    write_text(dongle / "idProduct", "520F\n");
    CHECK(plugins::avatar::detect_usb_device_presence(sysfs.path, 0x1915, 0x520F) == UsbDevicePresence::PRESENT);

    CHECK(plugins::avatar::detect_usb_device_presence(sysfs.path / "missing", 0x1915, 0x520F) ==
          UsbDevicePresence::UNKNOWN);
}

TEST_CASE("Dongle health distinguishes present, absent, and unavailable", "[sharpa_avatar][health]")
{
    const auto present = plugins::avatar::make_dongle_status(UsbDevicePresence::PRESENT);
    CHECK(present.state == core::PluginDeviceState_CONNECTED);
    CHECK(present.reason == core::PluginDeviceReason_NONE);

    const auto absent = plugins::avatar::make_dongle_status(UsbDevicePresence::ABSENT);
    CHECK(absent.state == core::PluginDeviceState_DISCONNECTED);
    CHECK(absent.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);

    const auto unknown = plugins::avatar::make_dongle_status(UsbDevicePresence::UNKNOWN);
    CHECK(unknown.state == core::PluginDeviceState_UNKNOWN);
    CHECK(unknown.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);
}
