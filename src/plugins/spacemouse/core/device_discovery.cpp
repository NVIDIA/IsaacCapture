// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "inc/spacemouse/device_discovery.hpp"

#include <algorithm>
#include <bitset>
#include <charconv>
#include <fstream>
#include <iterator>
#include <map>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <utility>
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
    //! Report layout when the node's report descriptor cannot be read.
    bool combined_report;
};

// Matched as substrings of HID_NAME, which is usually "<manufacturer> <product>", first match
// wins: list a name before any entry that is a substring of it.
constexpr KnownDevice kKnownDevices[] = {
    { "SpaceMouse Compact", false }, { "SpaceMouse Wireless BT", true },         { "SpaceMouse Wireless", false },
    { "SpaceNavigator", false },     { "3Dconnexion Universal Receiver", true },
};

// Generic Desktop X, Y, Z, Rx, Ry, Rz as extended usages (usage page << 16 | usage id).
constexpr uint32_t kFirstAxisUsage = 0x0001'0030;
constexpr std::size_t kAxisCount = 6;
constexpr uint8_t kTranslationReport = 1;
constexpr uint8_t kRotationReport = 2;

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

std::optional<std::vector<uint8_t>> read_bytes(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return std::nullopt;
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

// "hidraw10" after "hidraw9": a plain string sort puts it first.
std::pair<std::string_view, unsigned> numbered_name(std::string_view name)
{
    const auto digits = name.find_last_not_of("0123456789") + 1;
    unsigned number = 0;
    std::from_chars(name.data() + digits, name.data() + name.size(), number);
    return { name.substr(0, digits), number };
}

const KnownDevice* match_known_device(const std::string& hid_name)
{
    for (const auto& known : kKnownDevices)
    {
        if (hid_name.find(known.product_name) != std::string::npos)
            return &known;
    }
    return nullptr;
}

} // namespace

std::optional<bool> combined_report_from_descriptor(std::span<const uint8_t> descriptor)
{
    // Short items only matter here: Usage Page and Report ID (global), Usage and Usage Min/Max
    // (local), and Input (main), which assigns the pending usages to the current report.
    std::map<uint32_t, std::bitset<kAxisCount>> axes_by_report;
    uint32_t usage_page = 0;
    uint32_t report_id = 0;
    std::vector<std::pair<uint32_t, uint32_t>> globals_stack;
    std::vector<std::pair<uint32_t, uint32_t>> usage_ranges; // inclusive [min, max] extended usages
    std::optional<uint32_t> usage_minimum;

    std::size_t i = 0;
    while (i < descriptor.size())
    {
        const uint8_t prefix = descriptor[i];
        if (prefix == 0xFE) // long item: prefix, data size, tag, data
        {
            if (i + 1 >= descriptor.size())
                return std::nullopt;
            i += 3 + descriptor[i + 1];
            continue;
        }
        const std::size_t size = (prefix & 0x03) == 3 ? 4 : (prefix & 0x03);
        if (i + 1 + size > descriptor.size())
            return std::nullopt;
        uint32_t data = 0;
        for (std::size_t k = 0; k < size; ++k)
            data |= static_cast<uint32_t>(descriptor[i + 1 + k]) << (8 * k);
        i += 1 + size;

        const uint8_t type = (prefix >> 2) & 0x03;
        const uint8_t tag = prefix >> 4;
        // A 4-byte usage carries its own page; a shorter one takes the current Usage Page.
        const uint32_t extended_usage = size == 4 ? data : (usage_page << 16) | data;
        switch (type)
        {
        case 0: // main
            if (tag == 0x8) // Input
            {
                for (const auto& [low, high] : usage_ranges)
                {
                    for (std::size_t axis = 0; axis < kAxisCount; ++axis)
                    {
                        const uint32_t usage = kFirstAxisUsage + static_cast<uint32_t>(axis);
                        if (usage >= low && usage <= high)
                            axes_by_report[report_id].set(axis);
                    }
                }
            }
            usage_ranges.clear();
            usage_minimum.reset();
            break;
        case 1: // global
            if (tag == 0x0)
                usage_page = data;
            else if (tag == 0x8)
                report_id = data;
            else if (tag == 0xA)
                globals_stack.emplace_back(usage_page, report_id);
            else if (tag == 0xB && !globals_stack.empty())
            {
                std::tie(usage_page, report_id) = globals_stack.back();
                globals_stack.pop_back();
            }
            break;
        case 2: // local
            if (tag == 0x0)
                usage_ranges.emplace_back(extended_usage, extended_usage);
            else if (tag == 0x1)
                usage_minimum = extended_usage;
            else if (tag == 0x2 && usage_minimum)
                usage_ranges.emplace_back(*usage_minimum, extended_usage);
            break;
        default:
            break;
        }
    }

    const auto report_axes = [&](uint32_t id)
    {
        const auto it = axes_by_report.find(id);
        return it == axes_by_report.end() ? std::bitset<kAxisCount>{} : it->second;
    };
    const auto translation = report_axes(kTranslationReport);
    const auto rotation = report_axes(kRotationReport);
    if (translation.all())
        return true;
    if (translation == std::bitset<kAxisCount>(0b000111) && rotation == std::bitset<kAxisCount>(0b111000))
        return false;
    return std::nullopt;
}

std::optional<DeviceLocation> find_device(const std::filesystem::path& hidraw_class_dir,
                                          const std::filesystem::path& dev_dir)
{
    std::error_code ec;
    std::vector<std::string> names;
    // increment(ec), not range-for: operator++ throws when reading the next entry fails.
    for (std::filesystem::directory_iterator it(hidraw_class_dir, ec), end; !ec && it != end; it.increment(ec))
        names.push_back(it->path().filename().string());
    std::sort(names.begin(), names.end(),
              [](const std::string& a, const std::string& b) { return numbered_name(a) < numbered_name(b); });

    for (const auto& name : names)
    {
        const auto device_dir = hidraw_class_dir / name / "device";
        const auto hid_name = read_hid_name(device_dir / "uevent");
        const KnownDevice* known = hid_name ? match_known_device(*hid_name) : nullptr;
        if (!known)
            continue;
        const auto descriptor = read_bytes(device_dir / "report_descriptor");
        if (!descriptor)
            return DeviceLocation{ dev_dir / name, known->combined_report };
        if (const auto combined = combined_report_from_descriptor(*descriptor))
            return DeviceLocation{ dev_dir / name, *combined };
    }
    return std::nullopt;
}

} // namespace spacemouse
} // namespace plugins
