// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/deviceio_trackers/spacemouse_tracker.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <utility>
#include <vector>

namespace core
{

namespace
{

// Product names of the supported 3Dconnexion models. The Universal Receiver reports
// translation and rotation combined in a single 13-byte report; the others report them as two
// separate 7-byte reports.
struct KnownDevice
{
    std::string_view product_name;
    bool combined_report;
};

constexpr KnownDevice kKnownDevices[] = {
    { "SpaceMouse Compact", false },
    { "SpaceMouse Wireless", false },
    { "SpaceNavigator", false }, // also matches "SpaceNavigator for Notebooks
    { "3Dconnexion Universal Receiver", true },
};

// HID_NAME=<value> from a hidraw device's sysfs uevent file, or nullopt when unavailable.
std::optional<std::string> read_hid_name(const std::filesystem::path& uevent_path)
{
    std::ifstream file(uevent_path);
    if (!file.is_open())
        return std::nullopt;

    std::string line;
    while (std::getline(file, line))
    {
        constexpr std::string_view kPrefix = "HID_NAME=";
        if (line.starts_with(kPrefix))
            return line.substr(kPrefix.size());
    }
    return std::nullopt;
}

} // namespace

std::optional<SpaceMouseDevice> discover_spacemouse_device()
{
    const std::filesystem::path hidraw_class_dir = "/sys/class/hidraw";
    std::error_code ec;
    if (!std::filesystem::exists(hidraw_class_dir, ec))
        return std::nullopt;

    std::vector<std::string> hidraw_names;
    for (const auto& entry : std::filesystem::directory_iterator(hidraw_class_dir, ec))
        hidraw_names.push_back(entry.path().filename().string());
    std::sort(hidraw_names.begin(), hidraw_names.end());

    for (const auto& hidraw_name : hidraw_names)
    {
        const auto hid_name = read_hid_name(hidraw_class_dir / hidraw_name / "device" / "uevent");
        if (!hid_name)
            continue;

        for (const auto& known : kKnownDevices)
        {
            // HID_NAME is typically "<Manufacturer> <Product>"; match by substring so a
            // manufacturer prefix (e.g. "3Dconnexion SpaceMouse Compact") still matches.
            if (hid_name->find(known.product_name) != std::string::npos)
                return SpaceMouseDevice{ "/dev/" + hidraw_name, known.combined_report };
        }
    }
    return std::nullopt;
}

SpaceMouseTracker::SpaceMouseTracker(std::string device_path, bool combined_report)
    : device_path_(std::move(device_path)), combined_report_(combined_report)
{
}

const Serialized<SpaceMouseOutput>& SpaceMouseTracker::get_data(const ITrackerSession& session) const
{
    return static_cast<const ISpaceMouseTrackerImpl&>(session.get_tracker_impl(*this)).get_data();
}

} // namespace core
