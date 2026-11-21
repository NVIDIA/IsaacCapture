// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/spacemouse/device_discovery.hpp"

#include <algorithm>
#include <fstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace plugins
{
namespace spacemouse
{

namespace
{

struct KnownDevice
{
    std::string_view product_name;
    bool combined_report;
};

// Matched as substrings of HID_NAME, which is usually "<manufacturer> <product>".
// "SpaceNavigator" also matches "SpaceNavigator for Notebooks".
constexpr KnownDevice kKnownDevices[] = {
    { "SpaceMouse Compact", false },
    { "SpaceMouse Wireless", false },
    { "SpaceNavigator", false },
    { "3Dconnexion Universal Receiver", true },
};

std::optional<std::string> read_hid_name(const std::filesystem::path& uevent_path)
{
    std::ifstream file(uevent_path);
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

std::optional<DeviceLocation> find_device(const std::filesystem::path& hidraw_class_dir,
                                          const std::filesystem::path& dev_dir)
{
    std::error_code ec;
    std::vector<std::string> names;
    // increment(ec), not range-for: operator++ throws when reading the next entry fails.
    for (std::filesystem::directory_iterator it(hidraw_class_dir, ec), end; !ec && it != end; it.increment(ec))
        names.push_back(it->path().filename().string());
    std::sort(names.begin(), names.end());

    for (const auto& name : names)
    {
        const auto hid_name = read_hid_name(hidraw_class_dir / name / "device" / "uevent");
        if (!hid_name)
            continue;
        for (const auto& known : kKnownDevices)
        {
            if (hid_name->find(known.product_name) != std::string::npos)
                return DeviceLocation{ dev_dir / name, known.combined_report };
        }
    }
    return std::nullopt;
}

} // namespace spacemouse
} // namespace plugins
