// SPDX-FileCopyrightText: Copyright (c) 2026 Avatar SDK contributors. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <avatar_sdk/AvatarSDK.h>
#include <plugin_utils/plugin_device_status_publisher.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <optional>
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

struct PhysicalGloveHealthInput
{
    bool inventory_available = false;
    bool inventory_present = false;
    bool ever_observed = false;
    bool sdk_online = false;
    bool ingress_fresh = false;
    bool ever_received_ingress = false;
    std::optional<std::string> device_error;
};

struct DatasetHealthInput
{
    bool enabled = true;
    bool inventory_available = false;
    bool inventory_present = false;
    bool ever_observed = false;
    bool ingress_fresh = false;
    bool ever_received_ingress = false;
    bool data_fresh = false;
    bool ever_received_data = false;
    bool fetch_has_no_current_data = false;
    std::optional<std::string> fetch_error;
};

plugin_utils::PluginDeviceStatusEntry make_physical_glove_status(std::string path, const PhysicalGloveHealthInput& input);
plugin_utils::PluginDeviceStatusEntry make_dataset_status(std::string path, const DatasetHealthInput& input);
plugin_utils::PluginDeviceStatusEntry make_dongle_status(UsbDevicePresence presence, ::avatar::UsbConnectionState state);

UsbDevicePresence detect_usb_device_presence(const std::filesystem::path& devices_root,
                                             uint16_t vendor_id,
                                             uint16_t product_id);

} // namespace avatar
} // namespace plugins
