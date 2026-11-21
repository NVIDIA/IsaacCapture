// SPDX-FileCopyrightText: Copyright (c) 2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/avatar/avatar_health.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <system_error>
#include <utility>

namespace plugins
{
namespace avatar
{

namespace
{

std::optional<std::string> read_trimmed(const std::filesystem::path& path)
{
    std::ifstream stream(path);
    std::string value;
    if (!stream || !std::getline(stream, value))
    {
        return std::nullopt;
    }
    value.erase(
        std::remove_if(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c) != 0; }), value.end());
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
    return value;
}

std::string hex_id(uint16_t value)
{
    std::ostringstream stream;
    stream << std::hex << std::setfill('0') << std::setw(4) << value;
    return stream.str();
}

} // anonymous namespace

plugin_utils::PluginDeviceStatusEntry make_physical_glove_status(std::string path, const PhysicalGloveHealthInput& input)
{
    plugin_utils::PluginDeviceStatusEntry entry{ .path = std::move(path) };
    if (!input.inventory_available)
    {
        entry.state = core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_DEVICE_ERROR;
        entry.error = "Avatar SDK inventory is unavailable";
    }
    else if (!input.inventory_present)
    {
        entry.state = input.ever_observed ? core::PluginDeviceState_DISCONNECTED : core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
        entry.error = input.ever_observed ? "glove no longer appears in the Avatar SDK inventory" :
                                            "waiting for the Avatar SDK to discover the glove";
    }
    else if (input.device_error)
    {
        entry.state = core::PluginDeviceState_DEGRADED;
        entry.reason = core::PluginDeviceReason_DEVICE_ERROR;
        entry.error = *input.device_error;
    }
    else if (!input.ingress_fresh)
    {
        entry.state = input.ever_received_ingress ? core::PluginDeviceState_DEGRADED : core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_NO_CURRENT_DATA;
        entry.error = input.sdk_online ? "glove ingress is not advancing" : "Avatar SDK reports no recent glove ingress";
    }
    else
    {
        entry.state = core::PluginDeviceState_CONNECTED;
        entry.reason = core::PluginDeviceReason_NONE;
    }
    return entry;
}

plugin_utils::PluginDeviceStatusEntry make_dataset_status(std::string path, const DatasetHealthInput& input)
{
    plugin_utils::PluginDeviceStatusEntry entry{ .path = std::move(path) };
    if (!input.enabled)
    {
        entry.state = core::PluginDeviceState_DISABLED;
        entry.reason = core::PluginDeviceReason_DISABLED_BY_CONFIGURATION;
    }
    else if (!input.inventory_available)
    {
        entry.state = core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_DEVICE_ERROR;
        entry.error = "Avatar SDK inventory is unavailable";
    }
    else if (!input.inventory_present)
    {
        entry.state = input.ever_observed ? core::PluginDeviceState_DISCONNECTED : core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
        entry.error = input.ever_observed ? "glove no longer appears in the Avatar SDK inventory" :
                                            "waiting for the Avatar SDK to discover the glove";
    }
    else if (!input.ingress_fresh)
    {
        entry.state = input.ever_received_ingress ? core::PluginDeviceState_DEGRADED : core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_NO_CURRENT_DATA;
        entry.error = "waiting for current glove ingress";
    }
    else if (input.fetch_error)
    {
        entry.state = core::PluginDeviceState_DEGRADED;
        entry.reason = core::PluginDeviceReason_DEVICE_ERROR;
        entry.error = *input.fetch_error;
    }
    else if (input.fetch_has_no_current_data || !input.data_fresh)
    {
        entry.state = input.ever_received_data ? core::PluginDeviceState_DEGRADED : core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_NO_CURRENT_DATA;
        entry.error = input.ever_received_data ? "dataset is not advancing" : "waiting for the first dataset sample";
    }
    else
    {
        entry.state = core::PluginDeviceState_CONNECTED;
        entry.reason = core::PluginDeviceReason_NONE;
    }
    return entry;
}

plugin_utils::PluginDeviceStatusEntry make_dongle_status(UsbDevicePresence presence, ::avatar::UsbConnectionState state)
{
    plugin_utils::PluginDeviceStatusEntry entry{ .path = "/transport/dongle" };
    if (presence == UsbDevicePresence::UNKNOWN)
    {
        entry.state = core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
        entry.error = "USB device enumeration is unavailable";
        return entry;
    }
    if (presence == UsbDevicePresence::ABSENT)
    {
        entry.state = core::PluginDeviceState_DISCONNECTED;
        entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
        entry.error = "Avatar USB dongle 1915:520f is not enumerated";
        return entry;
    }

    switch (state)
    {
    case ::avatar::UsbConnectionState::CONNECTED:
        entry.state = core::PluginDeviceState_CONNECTED;
        entry.reason = core::PluginDeviceReason_NONE;
        break;
    case ::avatar::UsbConnectionState::DISCONNECTED:
        entry.state = core::PluginDeviceState_DISCONNECTED;
        entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
        entry.error = "Avatar SDK USB transport is disconnected";
        break;
    case ::avatar::UsbConnectionState::CONNECTING:
        entry.state = core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_NO_CURRENT_DATA;
        entry.error = "Avatar SDK USB transport is connecting";
        break;
    case ::avatar::UsbConnectionState::RECONNECTING:
        entry.state = core::PluginDeviceState_DEGRADED;
        entry.reason = core::PluginDeviceReason_NO_CURRENT_DATA;
        entry.error = "Avatar SDK USB transport is reconnecting";
        break;
    case ::avatar::UsbConnectionState::ERROR:
        entry.state = core::PluginDeviceState_FAILED;
        entry.reason = core::PluginDeviceReason_DEVICE_ERROR;
        entry.error = "Avatar SDK USB transport failed";
        break;
    }
    return entry;
}

UsbDevicePresence detect_usb_device_presence(const std::filesystem::path& devices_root,
                                             uint16_t vendor_id,
                                             uint16_t product_id)
{
    std::error_code error;
    std::filesystem::directory_iterator devices(devices_root, error);
    if (error)
    {
        return UsbDevicePresence::UNKNOWN;
    }

    const std::string expected_vendor = hex_id(vendor_id);
    const std::string expected_product = hex_id(product_id);
    bool incomplete_scan = false;
    const std::filesystem::directory_iterator end;
    for (; devices != end; devices.increment(error))
    {
        if (error)
        {
            return UsbDevicePresence::UNKNOWN;
        }
        const auto& device = *devices;
        // USB interface rows such as 1-2:1.0 do not own VID/PID files.
        if (device.path().filename().string().find(':') != std::string::npos)
        {
            continue;
        }
        const auto vendor = read_trimmed(device.path() / "idVendor");
        const auto product = read_trimmed(device.path() / "idProduct");
        if (!vendor || !product)
        {
            incomplete_scan = true;
            continue;
        }
        if (*vendor == expected_vendor && *product == expected_product)
        {
            return UsbDevicePresence::PRESENT;
        }
    }
    if (error)
    {
        return UsbDevicePresence::UNKNOWN;
    }
    return incomplete_scan ? UsbDevicePresence::UNKNOWN : UsbDevicePresence::ABSENT;
}

} // namespace avatar
} // namespace plugins
