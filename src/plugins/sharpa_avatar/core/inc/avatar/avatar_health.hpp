// SPDX-FileCopyrightText: Copyright (c) 2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <plugin_utils/plugin_device_status_publisher.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>

namespace plugins
{
namespace avatar
{

inline constexpr auto kAvatarDataTimeout = std::chrono::seconds(3);

enum class UsbDevicePresence
{
    UNKNOWN,
    ABSENT,
    PRESENT,
};

struct GloveHealthInput
{
    bool enabled = true;
    bool inventory_present = false;
    bool ever_observed = false;
    bool sdk_online = false;
    bool fresh_data = false;
    bool ever_received_fresh_data = false;
};

plugin_utils::PluginDeviceStatusEntry make_glove_status(std::string path, const GloveHealthInput& input);
plugin_utils::PluginDeviceStatusEntry make_dongle_status(UsbDevicePresence presence);

UsbDevicePresence detect_usb_device_presence(const std::filesystem::path& devices_root,
                                             uint16_t vendor_id,
                                             uint16_t product_id);

} // namespace avatar
} // namespace plugins
