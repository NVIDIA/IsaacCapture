// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/spacemouse/device_discovery.hpp"

#include <algorithm>
#include <cstdint>
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
// Match Wireless BT before Wireless: only the BT model uses combined reports.
// Keep one device per row so model ordering stays visible.
// clang-format off
constexpr KnownDevice kKnownDevices[] = {
    { "SpaceMouse Compact", false },
    { "SpaceMouse Wireless BT", true },
    { "SpaceMouse Wireless", false },
    { "SpaceNavigator", false },
    { "3Dconnexion Universal Receiver", true },
};
// clang-format on

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

bool has_motion_axes(const std::filesystem::path& descriptor_path)
{
    std::ifstream file(descriptor_path, std::ios::binary);
    uint32_t usage_page = 0;
    uint32_t usage_min = 0;
    unsigned usages = 0;
    unsigned input_axes = 0;
    // HID short items encode their payload size in the low two bits (3 means 4 bytes).
    for (int prefix; (prefix = file.get()) != EOF;)
    {
        if (prefix == 0xfe) // Long items are not used by the supported motion descriptors.
            return false;
        const int size = (prefix & 3) == 3 ? 4 : prefix & 3;
        uint32_t value = 0;
        for (int i = 0; i < size; ++i)
        {
            const int byte = file.get();
            if (byte == EOF)
                return false;
            value |= static_cast<uint32_t>(byte) << (8 * i);
        }
        switch (prefix & 0xfc)
        {
        case 0x04: // Usage Page
            usage_page = value;
            break;
        case 0x08: // Usage: Generic Desktop X/Y/Z/Rx/Ry/Rz are 0x30..0x35.
            if ((size == 4 ? value >> 16 : usage_page) == 1 && (value & 0xffff) >= 0x30 && (value & 0xffff) <= 0x35)
                usages |= 1U << ((value & 0xffff) - 0x30);
            break;
        case 0x18: // Usage Minimum
            usage_min = value;
            break;
        case 0x28: // Usage Maximum
            // Translation and rotation can use separate ranges and Input items.
            if (usage_page == 1)
                for (uint32_t usage = 0x30; usage <= 0x35; ++usage)
                    if (usage_min <= usage && usage <= value)
                        usages |= 1U << (usage - 0x30);
            break;
        case 0x80: // Input (ignore constant padding)
            if (!(value & 1))
                input_axes |= usages;
            if (input_axes == 0x3f)
                return true;
            break;
        }
        if ((prefix & 0x0c) == 0) // Local usages expire at every Main item.
        {
            usages = 0;
            usage_min = 0;
        }
    }
    return false;
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
            {
                // Receiver interfaces share HID_NAME; only the motion interface has these axes.
                if (known.product_name == "3Dconnexion Universal Receiver" &&
                    !has_motion_axes(hidraw_class_dir / name / "device" / "report_descriptor"))
                    continue;
                return DeviceLocation{ dev_dir / name, known.combined_report };
            }
        }
    }
    return std::nullopt;
}

} // namespace spacemouse
} // namespace plugins
