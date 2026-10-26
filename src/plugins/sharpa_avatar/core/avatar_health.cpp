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

plugin_utils::PluginDeviceStatusEntry make_glove_status(std::string path, const GloveHealthInput& input)
{
    plugin_utils::PluginDeviceStatusEntry entry{ .path = std::move(path) };
    if (!input.enabled)
    {
        entry.state = core::PluginDeviceState_DISABLED;
        entry.reason = core::PluginDeviceReason_DISABLED_BY_CONFIGURATION;
    }
    else if (!input.inventory_present)
    {
        entry.state = input.ever_observed ? core::PluginDeviceState_DISCONNECTED : core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
        entry.error = input.ever_observed ? "glove no longer appears in the Avatar SDK inventory" :
                                            "waiting for the Avatar SDK to discover the glove";
    }
    else if (!input.sdk_online)
    {
        entry.state = input.ever_received_fresh_data ? core::PluginDeviceState_DEGRADED : core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_NO_CURRENT_DATA;
        entry.error = "Avatar SDK reports online=false; the pinned SDK derives online from recent joint data";
    }
    else if (!input.fresh_data)
    {
        entry.state = input.ever_received_fresh_data ? core::PluginDeviceState_DEGRADED : core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_NO_CURRENT_DATA;
        entry.error =
            input.ever_received_fresh_data ? "glove has no fresh samples" : "waiting for the first fresh glove sample";
    }
    else
    {
        entry.state = core::PluginDeviceState_CONNECTED;
        entry.reason = core::PluginDeviceReason_NONE;
    }
    return entry;
}

plugin_utils::PluginDeviceStatusEntry make_dongle_status(UsbDevicePresence presence)
{
    plugin_utils::PluginDeviceStatusEntry entry{ .path = "/transport/dongle" };
    switch (presence)
    {
    case UsbDevicePresence::PRESENT:
        entry.state = core::PluginDeviceState_CONNECTED;
        entry.reason = core::PluginDeviceReason_NONE;
        break;
    case UsbDevicePresence::ABSENT:
        entry.state = core::PluginDeviceState_DISCONNECTED;
        entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
        entry.error = "Avatar USB dongle 1915:520f is not enumerated";
        break;
    case UsbDevicePresence::UNKNOWN:
        entry.state = core::PluginDeviceState_UNKNOWN;
        entry.reason = core::PluginDeviceReason_NO_HARDWARE_SIGNAL;
        entry.error = "USB device enumeration is unavailable";
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
    for (const auto& device : devices)
    {
        const auto vendor = read_trimmed(device.path() / "idVendor");
        const auto product = read_trimmed(device.path() / "idProduct");
        if (vendor && product && *vendor == expected_vendor && *product == expected_product)
        {
            return UsbDevicePresence::PRESENT;
        }
    }
    return UsbDevicePresence::ABSENT;
}

} // namespace avatar
} // namespace plugins
