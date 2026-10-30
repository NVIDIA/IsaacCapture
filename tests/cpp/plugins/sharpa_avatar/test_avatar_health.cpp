// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <avatar/avatar_hand_tracking_plugin.hpp>
#include <avatar/avatar_health.hpp>
#include <catch2/catch_test_macros.hpp>
#include <schema/serialized.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

using plugins::avatar::DatasetHealthInput;
using plugins::avatar::GloveIngressSignal;
using plugins::avatar::GloveState;
using plugins::avatar::HapticCommandDeliveryState;
using plugins::avatar::PhysicalGloveHealthInput;
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

TEST_CASE("Physical glove health follows inventory, ingress, and device errors", "[sharpa_avatar][health]")
{
    PhysicalGloveHealthInput input;

    SECTION("inventory failure is unknown rather than cached healthy")
    {
        input.inventory_present = true;
        input.ingress_fresh = true;
        const auto status = plugins::avatar::make_physical_glove_status("/glove/left", input);
        CHECK(status.state == core::PluginDeviceState_UNKNOWN);
        CHECK(status.reason == core::PluginDeviceReason_DEVICE_ERROR);
    }

    SECTION("startup before discovery")
    {
        input.inventory_available = true;
        const auto status = plugins::avatar::make_physical_glove_status("/glove/left", input);
        CHECK(status.state == core::PluginDeviceState_UNKNOWN);
        CHECK(status.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);
    }

    SECTION("previously observed glove disappears")
    {
        input.inventory_available = true;
        input.ever_observed = true;
        const auto status = plugins::avatar::make_physical_glove_status("/glove/left", input);
        CHECK(status.state == core::PluginDeviceState_DISCONNECTED);
        CHECK(status.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);
    }

    SECTION("online inventory row still waits for ingress")
    {
        input.inventory_available = true;
        input.inventory_present = true;
        input.sdk_online = true;
        const auto status = plugins::avatar::make_physical_glove_status("/glove/left", input);
        CHECK(status.state == core::PluginDeviceState_UNKNOWN);
        CHECK(status.reason == core::PluginDeviceReason_NO_CURRENT_DATA);
    }

    SECTION("fresh ingress is authoritative even if coarse SDK online lags")
    {
        input.inventory_available = true;
        input.inventory_present = true;
        input.ingress_fresh = true;
        input.ever_received_ingress = true;
        const auto status = plugins::avatar::make_physical_glove_status("/glove/left", input);
        CHECK(status.state == core::PluginDeviceState_CONNECTED);
        CHECK(status.reason == core::PluginDeviceReason_NONE);
    }

    SECTION("haptic or firmware errors degrade the physical glove")
    {
        input.inventory_available = true;
        input.inventory_present = true;
        input.ingress_fresh = true;
        input.ever_received_ingress = true;
        input.device_error = "set_vibration failed: CONNECTION_FAILED";
        const auto status = plugins::avatar::make_physical_glove_status("/glove/left", input);
        CHECK(status.state == core::PluginDeviceState_DEGRADED);
        CHECK(status.reason == core::PluginDeviceReason_DEVICE_ERROR);
        CHECK(status.error.find("set_vibration") != std::string::npos);
    }
}

TEST_CASE("Dataset health is independent per output", "[sharpa_avatar][health]")
{
    DatasetHealthInput input;

    SECTION("disabled output")
    {
        input.enabled = false;
        const auto status = plugins::avatar::make_dataset_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_DISABLED);
        CHECK(status.reason == core::PluginDeviceReason_DISABLED_BY_CONFIGURATION);
    }

    SECTION("inventory failure overrides cached data")
    {
        input.inventory_present = true;
        input.ingress_fresh = true;
        input.data_fresh = true;
        const auto status = plugins::avatar::make_dataset_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_UNKNOWN);
        CHECK(status.reason == core::PluginDeviceReason_DEVICE_ERROR);
    }

    SECTION("missing physical glove")
    {
        input.inventory_available = true;
        input.ever_observed = true;
        const auto status = plugins::avatar::make_dataset_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_DISCONNECTED);
        CHECK(status.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);
    }

    SECTION("fresh ingress without a first dataset sample remains startup")
    {
        input.inventory_available = true;
        input.inventory_present = true;
        input.ingress_fresh = true;
        input.ever_received_ingress = true;
        const auto status = plugins::avatar::make_dataset_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_UNKNOWN);
        CHECK(status.reason == core::PluginDeviceReason_NO_CURRENT_DATA);
    }

    SECTION("NO_VALID_DATA_RETURNED is no current data")
    {
        input.inventory_available = true;
        input.inventory_present = true;
        input.ingress_fresh = true;
        input.ever_received_ingress = true;
        input.fetch_has_no_current_data = true;
        const auto status = plugins::avatar::make_dataset_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_UNKNOWN);
        CHECK(status.reason == core::PluginDeviceReason_NO_CURRENT_DATA);
    }

    SECTION("other fetch failures surface as device errors")
    {
        input.inventory_available = true;
        input.inventory_present = true;
        input.ingress_fresh = true;
        input.ever_received_ingress = true;
        input.fetch_error = "Avatar SDK fetch_data failed: CONNECTION_FAILED";
        const auto status = plugins::avatar::make_dataset_status("/hand/left", input);
        CHECK(status.state == core::PluginDeviceState_DEGRADED);
        CHECK(status.reason == core::PluginDeviceReason_DEVICE_ERROR);
    }

    SECTION("each output keeps its own freshness")
    {
        input.inventory_available = true;
        input.inventory_present = true;
        input.ingress_fresh = true;
        input.ever_received_ingress = true;
        input.ever_received_data = true;

        input.data_fresh = true;
        const auto human = plugins::avatar::make_dataset_status("/hand/left", input);
        input.data_fresh = false;
        const auto raw = plugins::avatar::make_dataset_status("/glove/avatar_raw_left", input);

        CHECK(human.state == core::PluginDeviceState_CONNECTED);
        CHECK(raw.state == core::PluginDeviceState_DEGRADED);
        CHECK(raw.reason == core::PluginDeviceReason_NO_CURRENT_DATA);
    }
}

TEST_CASE("Ingress signals keep callbacks scoped to one device generation", "[sharpa_avatar][lifecycle]")
{
    GloveState state;
    auto retired = std::make_shared<GloveIngressSignal>();
    state.ingress = retired;
    retired->record();
    CHECK(retired->generation() == 1);
    CHECK(retired->last_event().has_value());

    state.reset();
    CHECK_FALSE(state.ingress);

    auto current = std::make_shared<GloveIngressSignal>();
    state.ingress = current;
    retired->record();
    CHECK(retired->generation() == 2);
    CHECK(current->generation() == 0);
    CHECK_FALSE(current->last_event().has_value());
}

TEST_CASE("Haptic delivery keeps failed commands pending until success", "[sharpa_avatar][haptic]")
{
    using namespace std::chrono_literals;

    core::HapticCommandT first_native;
    first_native.values = { 0.5f, 0.0f, 0.0f, 0.0f, 0.0f };
    const auto first = core::pack<core::HapticCommand>(first_native);
    const auto observed_at = HapticCommandDeliveryState::Clock::time_point{};

    HapticCommandDeliveryState delivery;
    delivery.observe(first, observed_at);
    REQUIRE(delivery.pending(observed_at, 200ms) == first.get());

    delivery.observe(first, observed_at + 100ms);
    CHECK(delivery.pending(observed_at + 100ms, 200ms) == first.get());

    delivery.acknowledge();
    CHECK(delivery.pending(observed_at + 100ms, 200ms) == nullptr);
    delivery.observe(first, observed_at + 100ms);
    CHECK(delivery.pending(observed_at + 100ms, 200ms) == nullptr);
}

TEST_CASE("Haptic delivery retains the newest command during retry backoff", "[sharpa_avatar][haptic]")
{
    using namespace std::chrono_literals;

    core::HapticCommandT first_native;
    first_native.values = { 0.5f, 0.0f, 0.0f, 0.0f, 0.0f };
    const auto first = core::pack<core::HapticCommand>(first_native);
    core::HapticCommandT second_native;
    second_native.values = { 0.0f, 0.5f, 0.0f, 0.0f, 0.0f };
    const auto second = core::pack<core::HapticCommand>(second_native);
    const auto observed_at = HapticCommandDeliveryState::Clock::time_point{};

    HapticCommandDeliveryState delivery;
    delivery.observe(first, observed_at);
    delivery.observe(second, observed_at + 150ms);
    CHECK(delivery.pending(observed_at + 250ms, 200ms) == second.get());

    delivery.discard();
    CHECK(delivery.pending(observed_at + 250ms, 200ms) == nullptr);
    delivery.observe(second, observed_at + 250ms);
    CHECK(delivery.pending(observed_at + 250ms, 200ms) == nullptr);
}

TEST_CASE("Haptic delivery expires commands that are no longer observed", "[sharpa_avatar][haptic]")
{
    using namespace std::chrono_literals;

    core::HapticCommandT command_native;
    command_native.values = { 0.5f, 0.0f, 0.0f, 0.0f, 0.0f };
    const auto command = core::pack<core::HapticCommand>(command_native);
    const auto observed_at = HapticCommandDeliveryState::Clock::time_point{};

    HapticCommandDeliveryState delivery;
    delivery.observe(command, observed_at);
    CHECK(delivery.pending(observed_at + 199ms, 200ms) == command.get());
    CHECK(delivery.pending(observed_at + 200ms, 200ms) == nullptr);

    delivery.observe(command, observed_at + 201ms);
    CHECK(delivery.pending(observed_at + 201ms, 200ms) == nullptr);
}

TEST_CASE("Dongle detection handles complete and incomplete USB enumeration", "[sharpa_avatar][health]")
{
    TemporaryDirectory sysfs;

    CHECK(plugins::avatar::detect_usb_device_presence(sysfs.path, 0x1915, 0x520F) == UsbDevicePresence::ABSENT);

    std::filesystem::create_directories(sysfs.path / "1-2:1.0");
    CHECK(plugins::avatar::detect_usb_device_presence(sysfs.path, 0x1915, 0x520F) == UsbDevicePresence::ABSENT);

    const auto incomplete_device = sysfs.path / "1-3";
    std::filesystem::create_directories(incomplete_device);
    write_text(incomplete_device / "idVendor", "1915\n");
    CHECK(plugins::avatar::detect_usb_device_presence(sysfs.path, 0x1915, 0x520F) == UsbDevicePresence::UNKNOWN);

    const auto dongle = sysfs.path / "1-2";
    std::filesystem::create_directories(dongle);
    write_text(dongle / "idVendor", "1915\n");
    write_text(dongle / "idProduct", "520F\n");
    CHECK(plugins::avatar::detect_usb_device_presence(sysfs.path, 0x1915, 0x520F) == UsbDevicePresence::PRESENT);

    CHECK(plugins::avatar::detect_usb_device_presence(sysfs.path / "missing", 0x1915, 0x520F) ==
          UsbDevicePresence::UNKNOWN);
}

TEST_CASE("Dongle health combines physical presence with the SDK link state", "[sharpa_avatar][health]")
{
    const auto connected =
        plugins::avatar::make_dongle_status(UsbDevicePresence::PRESENT, ::avatar::UsbConnectionState::CONNECTED);
    CHECK(connected.state == core::PluginDeviceState_CONNECTED);
    CHECK(connected.reason == core::PluginDeviceReason_NONE);

    const auto absent_with_stale_sdk_state =
        plugins::avatar::make_dongle_status(UsbDevicePresence::ABSENT, ::avatar::UsbConnectionState::CONNECTED);
    CHECK(absent_with_stale_sdk_state.state == core::PluginDeviceState_DISCONNECTED);
    CHECK(absent_with_stale_sdk_state.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);

    const auto unknown_presence =
        plugins::avatar::make_dongle_status(UsbDevicePresence::UNKNOWN, ::avatar::UsbConnectionState::CONNECTED);
    CHECK(unknown_presence.state == core::PluginDeviceState_UNKNOWN);
    CHECK(unknown_presence.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);

    const auto disconnected =
        plugins::avatar::make_dongle_status(UsbDevicePresence::PRESENT, ::avatar::UsbConnectionState::DISCONNECTED);
    CHECK(disconnected.state == core::PluginDeviceState_DISCONNECTED);
    CHECK(disconnected.reason == core::PluginDeviceReason_NO_HARDWARE_SIGNAL);

    const auto connecting =
        plugins::avatar::make_dongle_status(UsbDevicePresence::PRESENT, ::avatar::UsbConnectionState::CONNECTING);
    CHECK(connecting.state == core::PluginDeviceState_UNKNOWN);
    CHECK(connecting.reason == core::PluginDeviceReason_NO_CURRENT_DATA);

    const auto reconnecting =
        plugins::avatar::make_dongle_status(UsbDevicePresence::PRESENT, ::avatar::UsbConnectionState::RECONNECTING);
    CHECK(reconnecting.state == core::PluginDeviceState_DEGRADED);
    CHECK(reconnecting.reason == core::PluginDeviceReason_NO_CURRENT_DATA);

    const auto failed =
        plugins::avatar::make_dongle_status(UsbDevicePresence::PRESENT, ::avatar::UsbConnectionState::ERROR);
    CHECK(failed.state == core::PluginDeviceState_FAILED);
    CHECK(failed.reason == core::PluginDeviceReason_DEVICE_ERROR);
}
